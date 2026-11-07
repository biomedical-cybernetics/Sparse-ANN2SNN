from __future__ import annotations
import filecmp, json, os, random, tempfile
from pathlib import Path
from typing import Any

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

def configuration_data(value: Any) -> Any:
    """Convert configuration values to their JSON representation."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))

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


def preserve_checkpoint_link(source: str | Path, target: str | Path) -> None:
    """Create a checkpoint link, or retain an existing identical checkpoint."""
    source, target = Path(source), Path(target)
    try:
        os.link(source, target)
    except FileExistsError:
        filecmp.clear_cache()
        if target.is_file() and (source.samefile(target) or filecmp.cmp(source, target, shallow=False)):
            return
        raise ValueError(f'Existing final checkpoint differs from its source: {target}')
