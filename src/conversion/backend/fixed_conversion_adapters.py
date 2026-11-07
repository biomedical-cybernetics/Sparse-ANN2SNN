import copy, sys, types, os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

def upstream(kind):
    if kind != "parallel": raise ValueError("Unsupported converter")
    path=Path(os.environ.get("SUBMISSION_THIRD_PARTY_ROOT",Path(__file__).resolve().parents[3]/"third_party"/"sources"))/"parallel"/"modules.py"
    module=load_module("_fixed_parallel",path)
    module.ParaInfNeuron_CW_ND.forward=parallel_forward
    module.RecReLU.forward=recorded_relu_forward
    if not hasattr(module,"original_qcfs_forward"):module.original_qcfs_forward=module.QCFS.forward
    module.QCFS.forward=guarded_qcfs_forward
    return module


def load_module(name, path, replacements=(), injected=None):
    if name in sys.modules: return sys.modules[name]
    module = types.ModuleType(name); module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    module.__dict__.update(injected or {})
    text = Path(path).read_text()
    for old, new in replacements: text = text.replace(old, new)
    try: exec(compile(text, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None); raise
    return module


def annotate(tensor, events):
    tensor._conversion_events = events
    return tensor


def guard_zero_response(value,threshold):
    if not torch.isfinite(threshold).all() or torch.any(threshold<0): raise FloatingPointError("Invalid negative/nonfinite threshold")
    if torch.any((threshold==0)&(value>0)):
        raise FloatingPointError("Positive response in a zero-threshold channel; no automatic clipping")


def safe_denominator(threshold): return torch.where(threshold==0,torch.ones_like(threshold),threshold)


def fold_bn(model):
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            children = list(module._modules.items())
            for (name, conv), (bn_name, bn) in zip(children, children[1:]):
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    module._modules[name] = torch.nn.utils.fuse_conv_bn_eval(conv, bn)
                    module._modules[bn_name] = nn.Identity()
    return model


def parallel_forward(module, x):
    x = x.view((module.T, x.shape[0] // module.T) + x.shape[1:])
    response=module.TxT*x.mean(dim=0)+module.bias
    guard_zero_response(response,module.pre_th)
    events = ((response >= module.pre_th)&(module.pre_th>0)).float()
    return annotate((events * module.post_th).flatten(0, 1), events.flatten(0, 1).detach())


def recorded_relu_forward(module,x):
    # An all-negative preactivation has zero ReLU activity, not a negative clip
    # ceiling. Keep exact ANN ReLU outputs while recording their actual bounds.
    output=torch.relu(x)
    maximum=output.transpose(0,1).flatten(1).amax(1).unsqueeze(-1).unsqueeze(-1)
    if not module.init_up:
        module.init_up=True;module.register_buffer("up",maximum)
    else:module.up=torch.maximum(module.up,maximum)
    return output


def guarded_qcfs_forward(module,x):
    up=upstream("parallel")
    if not torch.any(module.up==0): return up.original_qcfs_forward(module,x)
    denominator=safe_denominator(module.up)
    if module.cab_inf:
        guard_zero_response(x+module.rec_in_mean,module.up)
        return torch.clamp(up.qcfs((x+module.rec_in_mean)*module.t/denominator+.5)/module.t,0,1)*(module.up+module.rec_th_mean)
    if module.is_cab:
        x=x.view((2,x.shape[0]//2)+x.shape[1:])
        guard_zero_response(x,module.up)
        err=(x[1]-x[0]).transpose(0,1).flatten(1)
        module.rec_in_mean=.99*module.rec_in_mean+.01*err.mean(1).unsqueeze(-1).unsqueeze(-1)
        guard_zero_response(x[0]+module.rec_in_mean,module.up)
        x[0]=torch.clamp(up.qcfs((x[0]+module.rec_in_mean)*module.t/denominator+.5)/module.t,0,1)*(module.up+module.rec_th_mean)
        x[1]=torch.clamp(x[1],torch.zeros_like(module.up),module.up)
        err=(x[1]-x[0]).transpose(0,1).flatten(1)
        module.rec_th_mean=.99*module.rec_th_mean+.01*err.mean(1).unsqueeze(-1).unsqueeze(-1)
        return x.flatten(0,1)
    guard_zero_response(x,module.up)
    return torch.clamp(up.qcfs(x/denominator*module.t+.5)/module.t,0,1)*module.up


def parallel_logits(model, x, time):
    output = model(x.unsqueeze(0).repeat(time, 1, 1, 1, 1).flatten(0, 1)).view(time, len(x), -1)
    if not torch.isfinite(output).all(): raise FloatingPointError("Nonfinite parallel logits")
    return [output[:i + 1].mean(0) for i in range(time)]


def finite_calibration(model):
    for name, module in model.named_modules():
        for field in ("threshold", "up", "pre_th", "post_th", "rec_th_mean", "rec_in_mean"):
            value = getattr(module, field, None)
            if torch.is_tensor(value):
                if not torch.isfinite(value).all(): raise FloatingPointError(f"Nonfinite {name}.{field}")
                if field in ("threshold", "up", "pre_th") and torch.any(value < 0):
                    indices=torch.where((value<0).reshape(value.shape[0],-1).any(1))[0].tolist()
                    raise FloatingPointError(f"Negative {name}.{field}; channels={indices}")
        if any(not torch.isfinite(v).all() for v in getattr(module, "time_dependent_biases", [])):
            raise FloatingPointError("Nonfinite temporal bias")
