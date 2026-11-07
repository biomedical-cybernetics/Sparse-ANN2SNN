"""Protect existing output files while coordinating a local torchrun launch."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time


def require_output(output, resume=False):
    output = Path(output)
    if resume:
        return
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Use an empty --output-dir, or supply --resume for recovery")
        return
    parent = os.getppid()
    try:
        stat = Path(f"/proc/{parent}/stat").read_text()
        parent_start = stat[stat.rfind(")") + 2:].split()[19]
    except OSError:
        parent_start = ""
    key = (f"{parent}-{parent_start or '0'}-"
           f"{os.environ.get('MASTER_PORT', '0')}-"
           f"{os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')}")
    directory = Path(tempfile.gettempdir()) / "sparse-ann-snn-launches"
    directory.mkdir(exist_ok=True)
    marker = directory / (key + ".json")
    if int(os.environ.get("RANK", "0")) == 0:
        error = None
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            error = "Use an empty --output-dir, or supply --resume for recovery"
        temporary = marker.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"output": str(output.resolve()), "error": error}))
        temporary.replace(marker)
    deadline = time.monotonic() + 30
    while True:
        if marker.is_file():
            checked = json.loads(marker.read_text())
            if checked.get("output") == str(output.resolve()):
                break
        if time.monotonic() >= deadline:
            raise ValueError("Timed out waiting for rank zero to check the output directory")
        time.sleep(.05)
    error = checked["error"]
    if error:
        raise ValueError(error)
