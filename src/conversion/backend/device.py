import os,random,copy
import numpy as np
import torch
from runtime import save_checkpoint_atomic

def configure(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def persist(path, model):
    value = copy.deepcopy(model).cpu()
    # Plain upstream tensor attributes are not registered buffers. Moving only
    # parameters misses thresholds, temporal biases and original ANN tensors.
    for module in value.modules():
        for name, item in list(vars(module).items()):
            if torch.is_tensor(item): setattr(module, name, item.detach().cpu())
            elif name == "time_dependent_biases": setattr(module, name, [v.detach().cpu() for v in item])
    save_checkpoint_atomic(path, value)


def device_model(model):
    model.cuda().eval()
    for module in model.modules():
        for name, item in list(vars(module).items()):
            if torch.is_tensor(item): setattr(module, name, item.cuda())
            elif name == "time_dependent_biases": setattr(module, name, [v.cuda() for v in item])
    return model
