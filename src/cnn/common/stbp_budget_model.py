import torch
from torch import nn
from torch.nn import functional as F

from Models.STBP_Model import MyCifarANNDense
from resnet_channel_chtss import ChannelCHTConv2d
from stbp_budget import CONV_INDICES
from table5_legacy.dst_scheduler import DSTScheduler


class FixedChannelConv(ChannelCHTConv2d):
    def __init__(self, ordinary, zero_pairs):
        nn.Module.__init__(self)
        for key in ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode"):
            setattr(self, key, getattr(ordinary, key))
        self.weight = ordinary.weight
        self.bias = ordinary.bias
        slots = ordinary.in_channels * ordinary.out_channels
        mask = torch.zeros(slots, dtype=torch.bool)
        mask[torch.randperm(slots)[:slots-zero_pairs]] = True
        self.register_buffer("mask", mask.reshape(ordinary.out_channels, ordinary.in_channels))
        self.register_buffer("evolve_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("soft_delta", torch.tensor(0.25, dtype=torch.float64))

    @torch.no_grad()
    def evolve(self):
        previous = self.mask.clone()
        active = int(previous.sum())
        count = int(0.05 * active)
        magnitude = self.weight.abs().mean(dim=(-1, -2))
        keep = self._soft_select(magnitude, ~self.mask, active-count, 1.0)
        self.mask.zero_(); self.mask.flatten()[keep] = True
        scores = self._ch3_scores()
        regrow = self._soft_select(scores, self.mask, count, float(self.soft_delta))
        self.mask.flatten()[regrow] = True
        if int(self.mask.sum()) != active:
            raise RuntimeError("Fixed channel density changed")
        self.evolve_count.add_(1)
        delta = float(self.soft_delta)
        self.soft_delta.fill_(min(0.75, delta + 0.01))
        return {"active": active, "requested_regrowth": count,
                "changed_pairs": int((previous != self.mask).sum()), "delta_used": delta,
                "update": int(self.evolve_count)}


def make_model(budget):
    model = MyCifarANNDense(budget["classes"], dropout=0.0)
    model.classifier.add_module("3", nn.Dropout(0.2))
    if budget["scheme"] == "channel-cht":
        for index, zeros in zip(CONV_INDICES, budget["conv_zero_weights"]):
            model.features[index] = FixedChannelConv(model.features[index], zeros // 9)
    return model


class BudgetLinearScheduler(DSTScheduler):
    def _current_prune_rate(self):
        # Power-of-two production Linear size makes the integer final target
        # exactly representable; avoid sigmoid endpoint floating-point drift.
        if self._current_prune_iter() == self.total_prune_iter:
            return float(self.args.sparsity)
        return super()._current_prune_rate()


def masks_for(model, pruner):
    masks = {f"features.{i}.weight": model.features[i].mask.detach().clone()
             for i in CONV_INDICES if isinstance(model.features[i], FixedChannelConv)}
    masks["classifier.1.weight"] = pruner.backward_masks[0].detach().clone()
    return masks


def budget_counts(model, pruner):
    conv = [(model.features[i].mask.numel()-int(model.features[i].mask.sum()))*9
            if isinstance(model.features[i], FixedChannelConv) else 0 for i in CONV_INDICES]
    mask = pruner.backward_masks[0]
    return conv, mask.numel()-int(mask.sum())


def qualified(model, pruner, budget):
    conv, linear = budget_counts(model, pruner)
    return conv == budget["conv_zero_weights"] and linear == budget["linear_zero_weights"]


def materialize(model, pruner):
    state = {name: value.detach().clone() for name, value in model.state_dict().items()
             if not name.endswith((".mask", ".evolve_count", ".soft_delta"))}
    for name, mask in masks_for(model, pruner).items():
        state[name].mul_(mask[:, :, None, None] if mask.ndim == 2 and state[name].ndim == 4 else mask)
    return state


@torch.no_grad()
def export_audit(model, pruner, budget, example):
    ordinary = MyCifarANNDense(budget["classes"], dropout=0.0).to(example.device)
    ordinary.classifier.add_module("3", nn.Dropout(0.2))
    ordinary.load_state_dict(materialize(model, pruner), strict=True)
    ordinary.eval(); model.eval()
    diff = float((ordinary(example) - model(example)).abs().max())
    if diff != 0:
        raise RuntimeError(f"Export changed logits: {diff}")
    conv, linear = budget_counts(model, pruner)
    macs = 0
    handles = []
    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, (nn.Conv2d, FixedChannelConv)):
            active = int(module.mask.sum())*9 if isinstance(module, FixedChannelConv) else module.weight.numel()
            macs += active*output.shape[-1]*output.shape[-2]
        elif isinstance(module, nn.Linear):
            macs += int(pruner.backward_masks[0].sum()) if module is model.classifier[1] else module.weight.numel()
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, FixedChannelConv, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model(example)
    for handle in handles: handle.remove()
    actual_conv_zeros = [model.features[i].weight.numel()-int(torch.count_nonzero(ordinary.features[i].weight))
                         for i in CONV_INDICES]
    actual_linear_zeros = ordinary.classifier[1].weight.numel()-int(torch.count_nonzero(ordinary.classifier[1].weight))
    actual_output_zeros = ordinary.last_layer.weight.numel()-int(torch.count_nonzero(ordinary.last_layer.weight))
    return {"logit_max_abs_difference": diff, "conv_zero_weights": conv,
            "linear_zero_weights": linear, "total_zero_weights": sum(conv)+linear,
            "weight_parameters": budget["total_weights"],
            "active_weight_parameters": budget["total_weights"]-sum(conv)-linear,
            "stored_parameters_including_bias_bn": sum(p.numel() for p in model.parameters()),
            "active_macs_per_sample": macs, "qualified": qualified(model, pruner, budget),
            "materialized_conv_zero_weights": actual_conv_zeros,
            "materialized_linear_zero_weights": actual_linear_zeros,
            "materialized_output_zero_weights": actual_output_zeros,
            "materialized_overall_sparsity": (sum(actual_conv_zeros)+actual_linear_zeros+actual_output_zeros)/budget["total_weights"],
            "overall_mask_sparsity": (sum(conv)+linear)/budget["total_weights"],
            "layer_mask_sparsity": [z/model.features[i].weight.numel() for i,z in zip(CONV_INDICES,conv)]
                + [linear/pruner.backward_masks[0].numel()]}
