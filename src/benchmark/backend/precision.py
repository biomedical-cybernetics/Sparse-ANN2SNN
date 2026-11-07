"""Precision helpers shared by dense and CHT training."""

from __future__ import annotations

from contextlib import AbstractContextManager

import torch
from timm.utils import NativeScaler, dispatch_clip_grad


AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def autocast_context(amp_dtype: str) -> AbstractContextManager:
    try:
        dtype = AMP_DTYPES[amp_dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported AMP dtype: {amp_dtype}") from exc
    return torch.amp.autocast("cuda", dtype=dtype)


class DisabledScaler:
    """NativeScaler-compatible unscaled optimizer step for BF16."""

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        clip_mode="norm",
        parameters=None,
        create_graph=False,
        need_update=True,
    ) -> None:
        loss.backward(create_graph=create_graph)
        if not need_update:
            return
        if clip_grad is not None:
            if parameters is None:
                raise ValueError("parameters are required when gradient clipping is enabled")
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        optimizer.step()

    def state_dict(self) -> dict[str, object]:
        return {"enabled": False, "amp_dtype": "bf16"}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        if state_dict and state_dict.get("amp_dtype") not in {None, "bf16"}:
            raise ValueError("cannot load a non-BF16 scaler state into BF16 training")


def build_loss_scaler(amp_dtype: str):
    if amp_dtype == "fp16":
        return NativeScaler()
    if amp_dtype == "bf16":
        return DisabledScaler()
    raise ValueError(f"unsupported AMP dtype: {amp_dtype}")
