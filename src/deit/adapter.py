"""Public dispatch for DeiT training, pruning and fixed-mask fine-tuning."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent


def validate_config(config):
    """Check that displayed settings and the selected source arguments agree."""
    p = config["parameters"]
    argv = p["arguments"]
    def option(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    if p["implementation"] == "finetune":
        expected = dict(epochs=21, world_size=8, per_gpu_batch_size=128,
                        global_batch_size=1024, warmup_epochs=5, sparsity=.7,
                        fixed_topology=True, base_learning_rate=.0005,
                        effective_learning_rate=.001, optimizer="AdamW",
                        optimizer_betas=[.9, .999], optimizer_epsilon=1e-8,
                        weight_decay=.05, minimum_learning_rate=1e-5,
                        scheduler="cosine", warmup_learning_rate=1e-6,
                        model_ema_decay=.99996, precision="fp16_amp", tf32=False)
        if any(p[k] != value for k, value in expected.items()):
            raise ValueError("Fixed-mask fine-tuning configuration differs from its training implementation")
    elif config["stage"] == "prune":
        if (float(option("--sparsity_ratio")) != p["sparsity"] or
                int(option("--nsamples")) != p["calibration_samples"] or
                int(option("--batch-size")) != p["per_gpu_batch_size"] or
                option("--prune_method") != p["method"] or
                ("--reconstruction" in argv) != p["reconstruction"]):
            raise ValueError("Pruning parameters disagree with source arguments")
    else:
        pairs = [("--epochs", "epochs", int), ("--batch-size", "per_gpu_batch_size", int),
                 ("--lr", "base_learning_rate", float), ("--sparsity", "sparsity", float),
                 ("--warmup-epochs", "warmup_epochs", int),
                 ("--weight-decay", "weight_decay", float)]
        if any(cast(option(flag)) != p[key] for flag, key, cast in pairs):
            raise ValueError("Training parameters disagree with source arguments")
        if p["global_batch_size"] != p["per_gpu_batch_size"] * p["world_size"]:
            raise ValueError("Global batch size is inconsistent")
        if p["effective_learning_rate"] != p["base_learning_rate"] * p["global_batch_size"] / 512:
            raise ValueError("Effective learning rate is inconsistent")
        constants = dict(optimizer_betas=[.9, .999], optimizer_epsilon=1e-8,
                         minimum_learning_rate=1e-5, warmup_learning_rate=1e-6,
                         model_ema_decay=.99996, precision="fp16_amp", pretrained=False)
        if any(p[k] != value for k, value in constants.items()):
            raise ValueError("Training parameters disagree with source defaults")
        if option("--opt").lower() != p["optimizer"].lower() or option("--sched") != p["scheduler"]:
            raise ValueError("Optimizer or scheduler disagrees with source arguments")


def _configuration(config, seed):
    return json.loads(json.dumps({"config": config, "seed": seed}, allow_nan=False))


def _world(config):
    expected = config["parameters"]["world_size"]
    actual = int(os.environ.get("WORLD_SIZE", "1"))
    if actual != expected:
        raise ValueError(f"This configuration requires torchrun --nproc_per_node={expected}; got {actual}")


def arguments(config, runtime):
    """Resolve source arguments without importing or executing a scientific module."""
    validate_config(config)
    p = config["parameters"]
    seed = int(runtime["seed"])
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    data = str(Path(runtime["data_root"]).resolve())
    output = str(Path(runtime["output_dir"]).resolve())
    argv = list(p["arguments"])
    if config["stage"] == "prune":
        inputs = runtime.get("inputs", {})
        checkpoint = runtime.get("checkpoint") or inputs.get("checkpoint") or inputs.get("dense_checkpoint")
        if not checkpoint:
            raise ValueError("Pruning requires --checkpoint from the matching Dense seed")
        argv += ["--ckpt", str(Path(checkpoint).resolve()), "--save_dir", output,
                 "--data-path", data, "--calib_dataset", str(Path(data) / p["calibration_split"]),
                 "--seed", str(seed)]
    elif p["implementation"] != "finetune":
        argv += ["--data-path", data, "--output_dir", output, "--seed", str(seed)]
        if runtime.get("resume"):
            resume = Path(runtime["resume"]).resolve()
            argv += ["--resume", str(resume)]
            if p["implementation"] == "cht":
                pruner = runtime.get("inputs", {}).get("pruner_checkpoint")
                if not pruner:
                    pruner = resume.with_name("best_pruner.pth" if resume.name == "best_checkpoint.pth" else "pruner.pth")
                argv += ["--pruner_statedict", str(pruner)]
    return argv


def execute(config, runtime):
    """Execute the selected stage."""
    validate_config(config)
    _world(config)
    p = config["parameters"]
    seed = int(runtime["seed"])
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    output = Path(runtime["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ANN2SNN_RUN_ID"] = f"{config['id']}-seed-{seed}"
    os.environ["ANN2SNN_CONFIGURATION"] = json.dumps(_configuration(config, seed), allow_nan=False)
    os.environ["ANN2SNN_WORLD_SIZE"] = str(p["world_size"])
    # Preserve the checkpoint deserialization behavior of the source implementation.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    if p["implementation"] == "finetune":
        return _finetune(config, runtime)
    version = "dense" if config["stage"] == "prune" else p["implementation"]
    source = HERE / version
    sys.path.insert(0, str(source))
    if config["stage"] == "prune":
        source = HERE / "pruning"
        sys.path.insert(0, str(source))
        script = source / "vit.py"
    else:
        script = source / "main.py"
        if version == "cht":
            os.environ["ANN2SNN_METHOD"] = "cht"
    sys.argv = [str(script), *arguments(config, runtime)]
    runpy.run_path(str(script), run_name="__main__")


def _finetune(config, runtime):
    source = HERE / "finetune"
    sys.path.insert(0, str(source))
    import torch
    import torch.distributed as dist
    from torchvision.datasets import ImageFolder
    from timm.models import create_model
    import main as legacy
    import models  # Register the model used by the fixed-mask implementation.
    from dst_util import recur_split_atten, get_W
    from fixed_topology import validate_masks
    import worker

    inputs = runtime.get("inputs", {})
    checkpoint = runtime.get("checkpoint") or inputs.get("checkpoint") or inputs.get("pruned_checkpoint")
    mask = inputs.get("mask")
    if not checkpoint or not mask:
        raise ValueError("Fine-tuning requires --checkpoint and --input mask=PATH")
    checkpoint, mask = str(Path(checkpoint).resolve()), str(Path(mask).resolve())
    data_root = Path(runtime["data_root"]).resolve()
    seed = int(runtime["seed"])
    args = legacy.get_args_parser().parse_args([
        "--epochs", "21", "--batch-size", "128", "--lr", "0.0005", "--sparsity", "0.7",
        "--fixed-topology", "--mask", mask, "--data-path", str(data_root), "--data-set", "IMNET", "--train-mode"])
    # These preparations precede train(), which seeds each rank before initialization.
    model = create_model(args.model, pretrained=False, num_classes=1000,
                         drop_rate=0., drop_path_rate=.1, img_size=224)
    recur_split_atten(model)
    masks = torch.load(mask, map_location="cpu", weights_only=False)
    if isinstance(masks, dict):
        masks = next(masks[k] for k in ("backward_masks", "masks", "mask") if k in masks)
    masks = [m.detach().bool().cpu().contiguous() for m in masks]
    weights, _, _ = get_W(model, args)
    budget = validate_masks(weights, masks, .7)
    del weights, model, masks
    references = {}
    for split in ("train", "test"):
        dataset = ImageFolder(data_root / split)
        references[split] = dict(samples=len(dataset), classes=dataset.class_to_idx)
    source_record = dict(checkpoint=checkpoint, mask=mask, budget=budget)
    manifest = dict(identity=_configuration(config, seed), data_root=str(data_root), smoke=False,
                    dataset_reference=references, inputs={f"sparsegpt-s{seed}": source_record})
    job = dict(id=f"sparsegpt-e21-s{seed}", method="sparsegpt", epochs=21, seed=seed, stage="training")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    worker.setup_precision(device)
    dist.init_process_group("nccl")
    try:
        worker.train(manifest, job, Path(runtime["output_dir"]).resolve(), runtime.get("resume"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
