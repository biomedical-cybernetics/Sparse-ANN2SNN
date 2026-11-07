#!/usr/bin/env python3
"""Run one configured experiment stage or inspect its configuration."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINES = {
    "cnn": "src.cnn.adapter",
    "deit": "src.deit.adapter",
    "direct": "src.direct.adapter",
    "conversion": "src.conversion.adapter",
    "benchmark": "src.benchmark.adapter",
}
STAGES = ("train", "prune", "convert", "evaluate", "benchmark")


def read_config(path):
    path = Path(path).expanduser()
    if not path.is_file():
        path = ROOT / path
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = ("id", "engine", "stage", "seeds", "paper", "parameters")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")
    if config["engine"] not in ENGINES or config["stage"] not in STAGES:
        raise ValueError("Unknown engine or stage")
    if not isinstance(config["seeds"], list) or not config["seeds"] or any(type(s) is not int for s in config["seeds"]):
        raise ValueError("seeds must be a nonempty list of integers")
    if len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("Duplicate seeds")
    if not isinstance(config["parameters"], dict):
        raise ValueError("parameters must be an object")
    module_path = ROOT.joinpath(*ENGINES[config["engine"]].split(".")).with_suffix(".py")
    if not module_path.is_file():
        raise ValueError(f"Missing execution module: {module_path.relative_to(ROOT)}")
    return config, path.resolve()


def named_inputs(values):
    result = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise ValueError("Use --input NAME=PATH")
        if key in result:
            raise ValueError(f"Duplicate input: {key}")
        result[key] = str(Path(value).expanduser().resolve())
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", nargs="?", choices=STAGES)
    p.add_argument("--config", help="JSON recipe path")
    p.add_argument("--list", action="store_true", help="List recipes without importing scientific packages")
    p.add_argument("--check-config", action="store_true", help="Validate and print a recipe without executing it")
    p.add_argument("--data-root")
    p.add_argument("--output-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--checkpoint")
    p.add_argument("--resume")
    p.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--gpus", help="Physical GPU IDs for the benchmark controller")
    p.add_argument("--nccl-library", help="Optional explicit NCCL library for benchmark subprocesses")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.list:
            if args.config or args.stage or args.check_config:
                raise ValueError("Use --list alone")
            for path in sorted((ROOT / "configs").rglob("*.json")):
                config, _ = read_config(path)
                print(f"{config['stage']:10} {path.relative_to(ROOT)}  seeds={config['seeds']}")
            return 0
        if not args.config:
            raise ValueError("--config is required")
        config, path = read_config(args.config)
        if args.stage and args.stage != config["stage"]:
            raise ValueError(f"This recipe is for {config['stage']}, not {args.stage}")
        if args.seed is not None and not 0 <= args.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        if config["engine"] == "benchmark" and args.seed is not None and args.seed not in config["seeds"]:
            raise ValueError(f"Select one of the configured seeds: {config['seeds']}")
        inputs = named_inputs(args.input)
        if args.check_config:
            adapter = importlib.import_module(ENGINES[config["engine"]])
            validate = getattr(adapter, "validate_config", None)
            if validate is not None:
                validate(config)
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0
        if not args.stage:
            raise ValueError("Specify a stage to execute, or use --check-config")
        if args.seed is None or not args.data_root or not args.output_dir:
            raise ValueError("Execution requires --seed, --data-root and --output-dir")
        for name in ("checkpoint", "resume"):
            value = getattr(args, name)
            if value:
                value = str(Path(value).expanduser().resolve())
                setattr(args, name, value)
                if name in inputs and inputs[name] != value:
                    raise ValueError(f"Conflicting {name} inputs")
                inputs[name] = value
        missing = [key for key in config.get("required_inputs", []) if key not in inputs]
        if missing:
            raise ValueError("Missing inputs: " + ", ".join(missing))
        for name, value in inputs.items():
            if not Path(value).exists():
                raise ValueError(f"Input {name} does not exist: {value}")
        data_root = Path(args.data_root).expanduser().resolve()
        if not data_root.is_dir():
            raise ValueError(f"Data directory does not exist: {data_root}")
        output = Path(args.output_dir).expanduser().resolve()
        world = int(os.environ.get("WORLD_SIZE", "1"))
        required_world = config.get("world_size", config["parameters"].get("world_size"))
        if required_world and config["engine"] != "benchmark" and world != required_world:
            raise ValueError(f"This recipe requires torchrun with {required_world} processes; WORLD_SIZE={world}")
        if config["engine"] == "benchmark" and world != 1:
            raise ValueError("Start the benchmark controller with python; it launches its own eight-rank workers")
        runtime = dict(data_root=str(data_root), output_dir=str(output), seed=args.seed,
                       checkpoint=args.checkpoint, resume=args.resume, inputs=inputs,
                       gpus=args.gpus, nccl_library=args.nccl_library, config_path=str(path))
        from src.output_guard import require_output
        require_output(output, resume=bool(args.resume))
        # Import the selected backend only after all side-effect-free checks.
        module = importlib.import_module(ENGINES[config["engine"]])
        module.execute(config, runtime)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
