import copy
from types import SimpleNamespace
import torch
from torch import nn
from conversion_pilot_adapters import install as upstream_install,threshold_floor,parallel_calibration_mode
from fixed_conversion_adapters import annotate,fold_bn,finite_calibration,parallel_forward
from DCGS.neurons import MTH_diff

def install(kind):
    if kind != "parallel": raise ValueError("Unsupported converter")
    value=upstream_install(kind)
    if not hasattr(value,"SummaryParaInfNeuron"):
        value.SummaryParaInfNeuron=type("SummaryParaInfNeuron",(value.ParaInfNeuron_CW_ND,),{"__module__":value.__name__,"forward":summary_parallel_forward})
    return value

def ensure_summaries(model,job):
    value=install(job.converter)
    for module in model.modules():
        if isinstance(module,value.ParaInfNeuron_CW_ND):module.__class__=value.SummaryParaInfNeuron
    if job.architecture=="ResNet-18":
        for module in model.conv1.modules():
            if isinstance(module,nn.ReLU):module.__class__=AnalogSummaryReLU;module.summary_slots=job.time
    return model


def summarize(output,slots=1,analog=False):
    events=(output!=0) if analog else getattr(output,"_conversion_events",None)
    if events is None:return output
    if len(events)%slots:raise ValueError("Event summary temporal layout mismatch")
    totals=events.reshape(slots,len(events)//slots,*events.shape[1:]).sum(1).detach().cpu()
    output._conversion_cpu_totals=totals.double()
    output._conversion_summary_slots=slots
    output._conversion_summary_analog=analog
    return output


def summary_parallel_forward(module,inputs):
    return summarize(parallel_forward(module,inputs),module.T)


class AnalogSummaryReLU(nn.ReLU):
    def forward(self,x):return summarize(super().forward(x),self.summary_slots,analog=True)


def root_input_summary(model,x,job):
    # The input photograph is shared across all encoded slots. Record its
    # analog nonzero proxy identically with and without the external monitor.
    counts=(x.detach().cpu()!=0).sum(0).double()
    slots=job.time+1 if job.is_dcgs else job.time if job.converter=="parallel" else 1
    values=counts.unsqueeze(0).repeat(slots,*([1]*counts.ndim))
    if job.is_dcgs:
        values.zero_();values[1]=counts
        values=values[1:]
    first=next(m for m in model.modules() if isinstance(m,nn.Conv2d) or hasattr(m,"fwd_func"))
    first._conversion_root_analog_totals=values


class TrackedMTHDiff(MTH_diff):
    """Unchanged multi-step DCGS equations plus generated signed event counts."""
    event_source = True

    def forward(self, x):
        if self.step_mode != "m": raise ValueError("This experiment uses DCGS multi-step mode only")
        time_steps = 2 ** torch.arange(self.num_thresholds, device=x.device)
        for _ in range(x.dim()): time_steps = time_steps.unsqueeze(-1)
        thre_shaped = self.thresh.unsqueeze(0) / time_steps
        thre_threshold = thre_shaped * .75
        x = self.expand(x)
        mem = torch.zeros_like(x[0]); spike_pot = [torch.zeros_like(x[0])]
        event_rows = [torch.zeros_like(x[0])]
        cumulative_out = torch.zeros_like(x[0]); bias = x[0].clone(); exp_in = x[0].clone()
        exp_out = torch.zeros_like(x[0])
        for t in range(self.T):
            mem = mem + x[t+1] - bias + exp_in - exp_out
            cumulative_out = cumulative_out + exp_out
            pos_false = (mem.unsqueeze(0) >= thre_threshold).float()
            neg_false = (mem.unsqueeze(0) <= -thre_threshold).float() * (cumulative_out.unsqueeze(0) >= thre_shaped).float()
            pos_true = pos_false.clone(); pos_true[1:] -= pos_false[:-1]
            neg_true = neg_false.clone(); neg_true[1:] -= neg_false[:-1]
            spike = torch.sum(thre_shaped * pos_true, dim=0) - torch.sum(thre_shaped * neg_true, dim=0)
            mem = mem - spike; cumulative_out = cumulative_out + spike
            exp_in = exp_in + (x[t+1] - bias)/(t+1); exp_out = exp_out + spike/(t+1)
            spike_pot.append(spike)
            # Count generated positive/negative threshold events, not a guess
            event_rows.append(((pos_true.abs()+neg_true.abs())*(thre_shaped!=0)).sum(0))
        out = self.merge(torch.stack(spike_pot,dim=0))
        events = self.merge(torch.stack(event_rows,dim=0))
        return summarize(annotate(out, events.detach()),self.T+1)


def track_dcgs_neurons(model):
    for module in model.modules():
        if isinstance(module,MTH_diff): module.__class__=TrackedMTHDiff
    return model


class EventMaxPool(nn.MaxPool2d):
    """Pool generated event counts through the unchanged per-channel max operation."""
    def forward(self, x):
        out = super().forward(x)
        events = getattr(x, "_conversion_events", None)
        if events is not None:
            if self.return_indices: raise ValueError("Unsupported indexed MaxPool event path")
            pooled = nn.functional.max_pool2d(events, self.kernel_size, self.stride, self.padding,
                self.dilation, self.ceil_mode, False)
            out = summarize(annotate(out, pooled.detach()),getattr(x,"_conversion_summary_slots",1))
        return out


def install_pools(model):
    for name, child in list(model.named_children()):
        if isinstance(child, nn.MaxPool2d) and not isinstance(child, EventMaxPool):
            if child.return_indices: raise ValueError("Indexed source MaxPool is unsupported")
            child.__class__ = EventMaxPool
        else: install_pools(child)
    return model


def replace_parallel(model, mode, job, prefix=""):
    up = install("parallel"); audits = []
    for name, child in list(model.named_children()):
        qualified = prefix+name
        if job.architecture == "ResNet-18" and qualified == "conv1":
            if mode=="spike":
                for layer in child.modules():
                    if isinstance(layer,nn.ReLU):layer.__class__=AnalogSummaryReLU;layer.summary_slots=job.time
            continue
        if mode == "record" and isinstance(child, nn.ReLU):
            setattr(model, name, up.RecReLU())
        elif mode == "calibrate" and isinstance(child, up.RecReLU):
            value, audit = threshold_floor(child.up*.5)
            module = up.PilotQCFS(in_ch=value.shape[0], up=value, t=job.time, is_cab=True, is_relu=True)
            module.pilot_gamma = .5; module.cab_inf = False; module.is_cab = True
            setattr(model,name,module)
            audits.append({"layer":qualified,"recorded":child.up.cpu().tolist(),"effective":value.cpu().tolist(),**audit})
        elif mode == "spike" and isinstance(child, up.QCFS):
            if torch.any(child.up+child.rec_th_mean < 0): raise FloatingPointError("Negative Parallel output amplitude")
            setattr(model,name,up.SummaryParaInfNeuron(job.time,child.up.cpu(),child.rec_th_mean.cpu(),
                (.5*child.up+child.rec_in_mean*job.time).cpu()))
        else: audits.extend(replace_parallel(child,mode,job,qualified+"."))
    return audits


def dcgs_args(job):
    return SimpleNamespace(architecture="VGG-16" if job.architecture=="VGG-16-GAP" else "ResNet-18",
        model_name="vgg16" if job.architecture=="VGG-16-GAP" else "resnet18", task="classification",
        mode="test_snn",threshold_mode="var",threshold_level="channel",neuron_name="MTH_diff",
        num_thresholds=4,time=job.time,step_mode="m",coding_type="diff_rate",fuse=False,tau=.98,
        dataset=job.dataset,bs=64,distributed=False,sop=True,seed=2026,dim=2)


def dcgs_model(ann, raw_calibration, job, device="cuda"):
    from DCGS.converter import Converter, Threshold_Getter
    from DCGS.forwards import forward_replace
    value = torch.load(raw_calibration,map_location="cpu",weights_only=False)
    if isinstance(value,nn.Module): model=value
    else:
        model=Converter.change_maxpool_before_relu(copy.deepcopy(ann))
        model=Threshold_Getter.replace_nonlinear_by_hook(model,momentum=.1,mode="var",level="channel")
        model.load_state_dict(value,strict=True)
    model=Converter.replace_by_maxpool_neuron(model,T=job.time,step_mode="m",coding_type="diff_rate")
    model.to(device).eval()
    Threshold_Getter.get_scale_from_var(model,T=job.time*16)
    args=dcgs_args(job);model=Converter("MTH_diff",args,T=job.time,step_mode="m",fuse_flag=False)(model)
    model=forward_replace(args,model).to(device).eval()
    finite_calibration(model)
    return track_dcgs_neurons(model)
