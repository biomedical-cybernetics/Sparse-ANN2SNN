import copy
import torch

def cpu_model(model):
    model = copy.deepcopy(model).cpu()
    for module in model.modules():
        # Calibration/monitor hooks are temporary closures, not model state.
        module._forward_hooks.clear(); module._forward_pre_hooks.clear()
        for name, value in list(vars(module).items()):
            if torch.is_tensor(value): setattr(module, name, value.detach().cpu())
            elif name == "time_dependent_biases": setattr(module, name, [v.detach().cpu() for v in value])
    return model
