import torch
from fixed_conversion_adapters import upstream

def install(kind):
    if kind != "parallel": raise ValueError("Unsupported converter")
    value=upstream(kind)
    if not hasattr(value,"PilotQCFS"):
        value.PilotQCFS=type("PilotQCFS",(value.QCFS,),{"__module__":value.__name__,"forward":pilot_qcfs_forward})
    return value


def threshold_floor(value):
    value = value.detach().float()
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise FloatingPointError("Negative/nonfinite threshold")
    positive = value[value > 0]
    floor = max(1e-6, 1e-4 * float(positive.median())) if positive.numel() else 1e-6
    zero = value == 0
    return torch.where(zero, torch.full_like(value, floor), value), {
        "floor": floor, "zero_indices": zero.nonzero().cpu().tolist(), "replaced": int(zero.sum()),
        "raw": value.cpu().tolist()}


def pilot_qcfs_forward(module, x):
    gamma = module.pilot_gamma
    if torch.any(module.up <= 0) or not torch.isfinite(module.up).all():
        raise FloatingPointError("Invalid DA-QCFS threshold")
    def quantize(value, shift=0., post=0.):
        return ((value+shift)*module.t/module.up+gamma).floor().div(module.t).clamp(0, 1)*(module.up+post)
    if module.cab_inf:
        return quantize(x, module.rec_in_mean, module.rec_th_mean)
    if module.is_cab:
        x = x.view((2, x.shape[0]//2)+x.shape[1:])
        # Avoid in-place modification of inputs shared by residual branches.
        student, teacher = x[0], x[1]
        error = (teacher-student).transpose(0, 1).flatten(1).mean(1).view(-1, 1, 1)
        module.rec_in_mean = .99*module.rec_in_mean+.01*error
        student = quantize(student, module.rec_in_mean, module.rec_th_mean)
        teacher = torch.minimum(teacher.clamp_min(0), module.up)
        error = (teacher-student).transpose(0, 1).flatten(1).mean(1).view(-1, 1, 1)
        module.rec_th_mean = .99*module.rec_th_mean+.01*error
        return torch.stack((student, teacher)).flatten(0, 1)
    return quantize(x)


def parallel_calibration_mode(model, enabled):
    up = install("parallel")
    for m in model.modules():
        if isinstance(m, up.QCFS): m.is_cab = enabled; m.cab_inf = not enabled
