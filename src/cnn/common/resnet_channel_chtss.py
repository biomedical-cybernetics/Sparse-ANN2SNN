from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

TARGET_SPARSITY = 0.78

INITIAL_SPARSITY = 0.50

SS_DURATION = 150

UPDATE_RATIO = 0.05

DELTA_INITIAL = 0.25

DELTA_MAX = 0.75

DELTA_INCREMENT = 0.01

class ChannelCHTConv2d(nn.Module):
    """Conv2d with one evolving binary link per output/input channel pair."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, *,
        stride: int = 1, padding: int = 0, bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (1, 1)
        self.groups = 1
        self.padding_mode = "zeros"
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

        slots = out_channels * in_channels
        active = slots - int(INITIAL_SPARSITY * slots)
        values = torch.rand(slots)
        indices = torch.topk(values, active).indices
        mask = torch.zeros(slots, dtype=torch.bool)
        mask[indices] = True
        self.register_buffer("mask", mask.view(out_channels, in_channels))
        self.register_buffer("evolve_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("current_sparsity", torch.tensor(INITIAL_SPARSITY, dtype=torch.float64))
        self.register_buffer("soft_delta", torch.tensor(DELTA_INITIAL, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked = self.weight * self.mask[:, :, None, None]
        return F.conv2d(x, masked, self.bias, self.stride, self.padding)

    @staticmethod
    def _soft_select(
        scores: torch.Tensor, omitted: torch.Tensor, count: int, delta: float,
    ) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        work = scores.clone()
        if delta < 0.95:
            work.pow_(delta / (1.0 - delta)).add_(1e-6)
            work[omitted] = 0.0
            return torch.multinomial(work.flatten(), count, replacement=False)
        work[omitted] = -1.0
        return torch.topk(work.flatten(), count).indices

    def _ch3_scores(self) -> torch.Tensor:
        adjacency = self.mask.float()
        transpose = adjacency.T
        dd = adjacency @ transpose
        tt = transpose @ adjacency
        bdd, btt = dd != 0, tt != 0
        elcl_dt = (adjacency.sum(dim=1) - dd) * bdd
        elcl_td = (transpose.sum(dim=1) - tt) * btt
        elcl_dt[elcl_dt == 0] = 1
        elcl_td[elcl_td == 0] = 1
        elcl_dt -= 1
        elcl_td -= 1
        elcl_dt = bdd / (elcl_dt + 1)
        elcl_td = btt / (elcl_td + 1)
        return elcl_dt @ adjacency + (elcl_td @ transpose).T

    @torch.no_grad()
    def evolve(self) -> dict[str, Any]:
        """Apply the recovered soft magnitude-removal/CH3-regrowth update."""
        slots = self.mask.numel()
        old_active = int(self.mask.sum().item())
        count = int(self.evolve_count.item())
        if count < SS_DURATION:
            exponent = -(count - SS_DURATION / 2)
            sparsity = INITIAL_SPARSITY + (TARGET_SPARSITY - INITIAL_SPARSITY) / (1 + math.exp(exponent))
            new_active = slots - int(sparsity * slots)
        else:
            sparsity = TARGET_SPARSITY
            new_active = slots - int(TARGET_SPARSITY * slots)
        regrow_count = int(UPDATE_RATIO * new_active)
        remove_count = old_active - (new_active - regrow_count)
        if not 0 <= remove_count <= old_active:
            raise RuntimeError(f"invalid channel removal count: {remove_count}/{old_active}")

        magnitude = self.weight.abs().mean(dim=(-1, -2))
        keep_count = old_active - remove_count
        keep = self._soft_select(magnitude, ~self.mask, keep_count, 1.0)
        self.mask.zero_()
        self.mask.view(-1)[keep] = True

        scores = self._ch3_scores()
        regrow = self._soft_select(scores, self.mask, regrow_count, float(self.soft_delta.item()))
        self.mask.view(-1)[regrow] = True
        if int(self.mask.sum().item()) != new_active:
            raise RuntimeError("channel topology update produced the wrong active-link count")
        self.evolve_count.add_(1)
        self.current_sparsity.fill_(1.0 - new_active / slots)
        self.soft_delta.fill_(min(DELTA_MAX, float(self.soft_delta.item()) + DELTA_INCREMENT))
        return {
            "update": count + 1,
            "active": new_active,
            "slots": slots,
            "sparsity": 1.0 - new_active / slots,
            "removed": remove_count,
            "regrown": regrow_count,
            "delta": float(self.soft_delta.item()),
        }
