from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_mask_file(path: str | Path) -> list[torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("backward_masks", "masks", "mask"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)):
        raise TypeError("fixed-topology mask must be a list or tuple of tensors")
    return [torch.as_tensor(mask).detach().bool().contiguous() for mask in payload]


def same_masks(left, right) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    return all(
        a is not None and b is not None and a.shape == b.shape
        and torch.equal(a.detach().bool().cpu(), b.detach().bool().cpu())
        for a, b in zip(left, right)
    )


def validate_masks(
    weights: list[torch.Tensor],
    masks: list[torch.Tensor],
    expected_sparsity: float,
) -> dict[str, Any]:
    if len(weights) != len(masks):
        raise ValueError(f"mask count mismatch: {len(masks)} masks for {len(weights)} weights")
    total = 0
    active = 0
    for index, (weight, mask) in enumerate(zip(weights, masks)):
        if tuple(weight.shape) != tuple(mask.shape):
            raise ValueError(
                f"mask {index} shape {tuple(mask.shape)} does not match "
                f"weight shape {tuple(weight.shape)}"
            )
        total += mask.numel()
        active += int(mask.sum().item())
    sparsity = 1.0 - active / total
    if abs(sparsity - expected_sparsity) > 1e-5:
        raise ValueError(
            f"mask sparsity {sparsity:.9f} differs from expected {expected_sparsity:.9f}"
        )
    return {
        "mask_count": len(masks),
        "active": active,
        "total": total,
        "sparsity": sparsity,
    }


@torch.no_grad()
def audit_fixed_topology(
    weights: list[torch.Tensor],
    masks: list[torch.Tensor],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    masked_weight_nonzero = 0
    masked_moment_nonzero = 0
    for weight, cpu_mask in zip(weights, masks):
        mask = cpu_mask.to(weight.device)
        masked_weight_nonzero += int(torch.count_nonzero(weight.detach()[~mask]).item())
        if optimizer is None:
            continue
        state = optimizer.state.get(weight, {})
        for key in ("exp_avg", "exp_avg_sq", "prev_grad", "prev_u", "e_w"):
            value = state.get(key)
            if torch.is_tensor(value) and value.shape == weight.shape:
                masked_moment_nonzero += int(torch.count_nonzero(value.detach()[~mask]).item())
    return {
        "masked_weight_nonzero": masked_weight_nonzero,
        "masked_moment_nonzero": masked_moment_nonzero,
    }
