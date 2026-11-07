import torch
from torch import nn
import cnn_fixed_conversion_adapters as base
from fixed_conversion_adapters import annotate

def install(kind):
    if kind != "parallel": raise ValueError("Unsupported converter")
    value=base.install(kind)
    if not hasattr(value,"CompletionRecReLU"):
        for name,parent,forward in (("CompletionRecReLU",value.RecReLU,rank2_record_forward),("CompletionQCFS",value.PilotQCFS,rank2_record_forward),("CompletionParaNeuron",value.SummaryParaInfNeuron,rank2_neuron_forward)):
            setattr(value,name,type(name,(parent,),{"__module__":value.__name__,"forward":forward,"_completion_base_forward":parent.forward}))
    return value


def rank2_record_forward(module,x):
    linear=x.ndim==2
    out=module._completion_base_forward(x[:,:,None,None] if linear else x)
    return out.squeeze(-1).squeeze(-1) if linear else out


def rank2_neuron_forward(module,x):
    linear=x.ndim==2
    out=module._completion_base_forward(x[:,:,None,None] if linear else x)
    if not linear:return out
    events=out._conversion_events.squeeze(-1).squeeze(-1)
    return base.summarize(annotate(out.squeeze(-1).squeeze(-1),events),module.T)


def replace_parallel(model,mode,job,prefix=""):
    up=install("parallel");audits=[]
    for name,child in list(model.named_children()):
        qualified=prefix+name
        if mode=="record" and isinstance(child,nn.ReLU):setattr(model,name,up.CompletionRecReLU())
        elif mode=="calibrate" and isinstance(child,up.RecReLU):
            value,audit=base.threshold_floor(child.up*.5)
            m=up.CompletionQCFS(in_ch=value.shape[0],up=value,t=job.time,is_cab=True,is_relu=True)
            m.pilot_gamma=.5;m.cab_inf=False;m.is_cab=True;setattr(model,name,m)
            audits.append({"layer":qualified,"recorded":child.up.cpu().tolist(),"effective":value.cpu().tolist(),**audit})
        elif mode=="spike" and isinstance(child,up.QCFS):
            if torch.any(child.up+child.rec_th_mean<0):raise FloatingPointError("Negative Parallel output amplitude")
            setattr(model,name,up.CompletionParaNeuron(job.time,child.up.cpu(),child.rec_th_mean.cpu(),(.5*child.up+child.rec_in_mean*job.time).cpu()))
        else:audits.extend(replace_parallel(child,mode,job,qualified+"."))
    return audits


def ensure_summaries(model,job):
    install(job.converter)
    # Completion classes already emit exact summaries for both ranks.
    return model
