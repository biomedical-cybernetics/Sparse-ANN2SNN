import torch
from torch import nn

@torch.no_grad()
def apply_masks(model, masks):
    for name, mask in masks.items():
        p = dict(model.named_parameters())[name]
        expanded = mask[:, :, None, None] if p.ndim == 4 and mask.ndim == 2 else mask
        expanded=expanded.to(p.device)
        p.mul_(expanded)
        if torch.count_nonzero(p[~expanded.bool().expand_as(p)]): raise ValueError("Fixed mask violated")
