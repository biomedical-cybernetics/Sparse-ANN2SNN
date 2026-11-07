import torch.nn as nn
from .load_data_imgnet import load_data_imagenet

def count_actiavtions(model : nn.Module):
    count = 0
    for mod in model.modules():
        if isinstance(mod, (nn.GELU, nn.ReLU)):
            count += 1
    return count
