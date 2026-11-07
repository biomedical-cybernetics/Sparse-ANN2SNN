from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def capture_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_checkpoint_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    import torch
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_checkpoint_checked(
    path: str | Path,
    *,
    run_id: str,
    configuration: dict[str, Any],
    seed: int,
    world_size: int,
) -> dict[str, Any]:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "run_id": run_id,
        "configuration": configuration,
        "seed": seed,
        "world_size": world_size,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume checkpoint identity mismatch: {mismatches}")
    if "epoch" not in payload or "rng_state" not in payload:
        raise ValueError("resume checkpoint lacks epoch or RNG state")
    return payload


def limited_batches(loader: Iterable[Any], maximum: int | None) -> Iterator[Any]:
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield batch
