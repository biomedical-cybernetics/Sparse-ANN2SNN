"""Public dispatch for DPAP and SD-SNN CIFAR training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent


def validate_config(config):
    p = config["parameters"]
    constants = dict(optimizer="AdamW", weight_decay=.01, scheduler="cosine",
                     minimum_learning_rate=1e-5, warmup_epochs=5,
                     warmup_learning_rate=1e-6, cooldown_epochs=10, world_size=1)
    if any(p[key] != expected for key, expected in constants.items()):
        raise ValueError("Direct-SNN configuration differs from source optimizer/scheduler settings")
    if p["effective_learning_rate"] != p["base_learning_rate"] * p["batch_size"] / 1024:
        raise ValueError("Effective learning rate is inconsistent")


def arguments(config, runtime):
    """Construct the selected training command without running it."""
    validate_config(config)
    p = config["parameters"]
    seed = int(runtime["seed"])
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    args = ["--dataset", p["dataset"], "--data-root", str(Path(runtime["data_root"]).resolve()),
            "--output-dir", str(Path(runtime["output_dir"]).resolve()), "--seed", str(seed),
            "--device", "0", "--epochs", str(p["epochs"]), "--batch-size", str(p["batch_size"]),
            "--lr", str(p["base_learning_rate"]), "--timesteps", str(p["timesteps"]),
            "--checkpoint-interval", str(p["checkpoint_interval"])]
    if runtime.get("resume"):
        args += ["--resume", str(Path(runtime["resume"]).resolve())]
    return args


def execute(config, runtime):
    validate_config(config)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Direct SNN training uses one GPU per run")
    p = config["parameters"]
    source = HERE / p["implementation"]
    identity = dict(config=config, seed=int(runtime["seed"]))
    os.environ["ANN2SNN_RUN_ID"] = f"{config['id']}-seed-{runtime['seed']}"
    os.environ["ANN2SNN_CONFIGURATION"] = json.dumps(identity, allow_nan=False)
    os.environ["ANN2SNN_WORLD_SIZE"] = "1"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path[:0] = [str(source), str(HERE)]
    script = source / "main_cifar10.py"
    sys.argv = [str(script), *arguments(config, runtime)]
    runpy.run_path(str(script), run_name="__main__")
