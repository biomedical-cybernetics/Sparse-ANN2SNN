from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from types import SimpleNamespace
import torch
from training_options import training_context
from torch import nn
import Models
from runtime import atomic_json, capture_rng_state, restore_rng_state, save_checkpoint_atomic, configuration_data
from stbp_c10_ablation import Job
from stbp_c10_ablation import checkpoint_epoch, cutoff_reference
from stbp_c10_ablation_model import make_model, FrozenBudgetLinearScheduler as BudgetLinearScheduler
from stbp_budget_model import FixedChannelConv, budget_counts, export_audit, masks_for, materialize, qualified
from vgg_hybrid_train_worker import evaluate, seed_all

def runtime_args(job, budget):
    return SimpleNamespace(architecture="STBPANN", dataset=job.dataset, bs=job.batch,
        dim=1, conv_sparsity=0., linear_sparsity=budget["linear_sparsity"],
        sparsity=budget["linear_sparsity"], dropout=0., one_fc=False, pruning="",
        remove_method="weight_magnitude_soft", regrow_method="CH3_L3n_soft",
        granet=True, gmp=False, granet_init_sparsity=.2, granet_init_step=0,
        pruning_T_end=None, pruning_scheduler="s_shape", pruning_method="ri", k=6.,
        sparsity_distribution="uniform", static_topo=False, adaptive_zeta=True,
        chain_removal=True, early_stop=False, EM_S=False, itop=False,
        history_weights=False, new_history_weights=False, init_mode="xavier", WS=False,
        BRF=True, brf_r=.1, degree_dist="uniform", ssam=False, factor=.01,
        start_T=1., end_T=1., delta=.5, delta_max=.75, delta_d=.01, delta_schedule="linear")

def save_best(model, pruner, artifacts, prefix):
    save_checkpoint_atomic(artifacts / f"{prefix}_model.pth", materialize(model, pruner))
    save_checkpoint_atomic(artifacts / f"{prefix}_masks.pth", masks_for(model, pruner))
    save_checkpoint_atomic(artifacts / f"{prefix}_pruner.pth", pruner.state_dict())

def best_snapshots(artifacts):
    """Bundle the exact best files belonging to the resumable epoch."""
    return {path.name: torch.load(path, map_location="cpu", weights_only=False)
            for prefix in ("best", "unrestricted_best")
            for suffix in ("model", "masks", "pruner")
            if (path := artifacts / f"{prefix}_{suffix}.pth").is_file()}

def restore_best_snapshots(state, artifacts, metadata):
    for name, value in state["best_artifacts"].items():
        save_checkpoint_atomic(artifacts / name, value)
    if state.get("best_export_audit"):
        atomic_json(metadata / "best_export_audit.json", state["best_export_audit"])

def train(args):
    config = json.loads(args.config.read_text())
    job = Job(**config["job"])
    budget, configuration = config["budget"], config["configuration"]
    expected = job.config({"groups": {"CIFAR10-sd": {"budgets": {job.scheme: budget}}}})
    if configuration_data(expected) != configuration:
        raise ValueError("Worker configuration mismatch")
    if args.probe_batches is not None and not 1 <= args.probe_batches <= 2:
        raise ValueError("Smoke is bounded to two batches")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    seed_all(job.seed)
    rargs = runtime_args(job, budget)
    unused, _, train_loader, test_loader = Models.prepare_model_and_loader(rargs, str(args.data_root))
    del unused
    if args.probe_batches is not None:
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=job.batch, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=job.batch, shuffle=False, num_workers=0)
    model = make_model(budget, job.dropout).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=job.lr, momentum=.9, weight_decay=5e-4, nesterov=True)
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=595, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cosine], milestones=[5])
    steps = len(train_loader)
    pruner = BudgetLinearScheduler(model, optimizer, alpha=.3, delta=steps, T_end=job.horizon*steps,
        sparsity_distribution="uniform", static_topo=False, ignore_linear_layers=False,
        grad_accumulation_n=1, args=rargs)
    if len(pruner.W) != 1 or tuple(pruner.W[0].shape) != (512, 32768):
        raise ValueError("Expected only the 32768-to-512 hidden Linear under DST")
    start_epoch = updates = optimizer_steps = 0
    best = {"top1": -1., "loss": None, "epoch": None}
    unrestricted = dict(best)
    history = []
    cutoff_checkpoint = args.artifact_dir / f"checkpoints/epoch-{job.horizon:03d}.pth"
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get('configuration') != configuration or state["job"] != job.__dict__:
            raise ValueError("Resume identity mismatch")
        if not checkpoint_epoch(state["epoch"], job.horizon):
            raise ValueError("Incomplete resume epoch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        pruner.load_state_dict(state["pruner"])
        restore_rng_state(state["rng"])
        start_epoch, updates, optimizer_steps = state["epoch"], state["updates"], state["optimizer_steps"]
        best, unrestricted, history = state["best"], state["unrestricted"], state["history"]
        restore_best_snapshots(state, args.artifact_dir, args.metadata_dir)
        if start_epoch >= job.horizon:
            cutoff_checkpoint = cutoff_reference(state, args.resume, job.horizon)
            cutoff = torch.load(cutoff_checkpoint, map_location="cpu", weights_only=False)
            if cutoff["epoch"] != job.horizon or cutoff["job"] != job.__dict__ or cutoff.get('configuration') != configuration:
                raise ValueError("Permanent evolution-cutoff checkpoint mismatch")
            del cutoff
        del state
    criterion = nn.CrossEntropyLoss()
    started = time.time()
    for epoch in range(start_epoch, 600):
        model.train(); correct = total = batches = 0; loss_sum = 0.
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        for index, (images, labels) in enumerate(train_loader):
            if args.probe_batches is not None and index >= args.probe_batches: break
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            with training_context(job):
                output = model(images); loss = criterion(output, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite ANN loss")
            loss.backward()
            if args.probe_batches is not None and updates == 0:
                pruner.step = pruner.delta_T-1
            previous = pruner.dst_steps
            if pruner(): optimizer.step(); optimizer_steps += 1
            updates += pruner.dst_steps-previous
            correct += int((output.argmax(1) == labels).sum()); total += labels.numel()
            loss_sum += float(loss); batches += 1
        if not batches: raise ValueError("Empty training loader")
        if args.probe_batches:
            # In addition to the nonzero-rewiring update above, exercise the
            # exact final budget and frozen clock without extra data batches.
            pruner.step = pruner.T_end-1
            previous = pruner.dst_steps
            pruner()
            updates += pruner.dst_steps-previous
            frozen_step = pruner.step
            assert pruner() is True and pruner.step == frozen_step
        scheduler.step()
        val, val_loss = evaluate(model, test_loader, torch.device("cuda:0"), 1 if args.probe_batches else None)
        at_budget = qualified(model, pruner, budget)
        row = {"epoch": epoch+1, "lr": epoch_lr, "train_top1": correct/total,
            "train_loss": loss_sum/batches, "test_top1": val, "test_loss": val_loss,
            "budget_qualified": at_budget, "conv_zero_weights": budget_counts(model, pruner)[0],
            "linear_zero_weights": budget_counts(model, pruner)[1],
            "linear_step": pruner.step, "linear_dst_steps": pruner.dst_steps,
            "linear_horizon": job.horizon, "hidden_dropout": job.dropout}
        if val > unrestricted["top1"]:
            unrestricted = {"top1": val, "loss": val_loss, "epoch": epoch+1}
            save_best(model, pruner, args.artifact_dir, "unrestricted_best")
        if at_budget and val > best["top1"]:
            best = {"top1": val, "loss": val_loss, "epoch": epoch+1}
            save_best(model, pruner, args.artifact_dir, "best")
            rng = capture_rng_state()
            example = next(iter(test_loader))[0][:1].cuda()
            audit = export_audit(model, pruner, budget, example)
            restore_rng_state(rng)
            atomic_json(args.metadata_dir / "best_export_audit.json", audit)
        events = [m.evolve() for m in model.modules() if isinstance(m, FixedChannelConv)]
        row["conv_updates"] = events
        history.append(row)
        print(json.dumps(row), flush=True)
        completed = epoch+1
        if checkpoint_epoch(completed, job.horizon) or args.probe_batches:
            payload = {"configuration": configuration, "job": job.__dict__, "epoch": completed,
                "cutoff_checkpoint": str(cutoff_checkpoint.resolve()),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "pruner": pruner.state_dict(),
                "rng": capture_rng_state(), "best": best, "unrestricted": unrestricted,
                "history": history, "updates": updates, "optimizer_steps": optimizer_steps,
                "best_artifacts": best_snapshots(args.artifact_dir),
                "best_export_audit": json.loads((args.metadata_dir / "best_export_audit.json").read_text())
                    if (args.metadata_dir / "best_export_audit.json").is_file() else None}
            save_checkpoint_atomic(args.artifact_dir / "checkpoints/last_checkpoint.pth", payload)
            save_checkpoint_atomic(args.artifact_dir / f"checkpoints/epoch-{completed:03d}.pth", payload)
        if args.probe_batches: break
    save_best(model, pruner, args.artifact_dir, "final")
    if not args.probe_batches and (len(history) != 600 or updates != job.horizon or not qualified(model, pruner, budget)):
        raise ValueError("Incomplete epochs, topology schedule, or final budget")
    if not args.probe_batches and best["epoch"] is None:
        raise ValueError("No budget-qualified checkpoint")
    rng = capture_rng_state()
    audit = export_audit(model, pruner, budget, next(iter(test_loader))[0][:1].cuda())
    restore_rng_state(rng)
    metrics = {"job": job.__dict__, "configuration": configuration, "budget": budget,
        "best_top1": best["top1"], "best_loss": best["loss"], "best_epoch": best["epoch"],
        "unrestricted_best": unrestricted, "history": history, "linear_updates": updates,
        "optimizer_steps": optimizer_steps, "elapsed_seconds": time.time()-started,
        "probe": bool(args.probe_batches), "export_audit": audit,
        "conv_updates_per_layer": [int(m.evolve_count) for m in model.modules() if isinstance(m, FixedChannelConv)],
        "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}
    metrics["cutoff_checkpoint"] = str(cutoff_checkpoint.resolve())
    if not metrics["all_parameters_finite"]: raise FloatingPointError("Nonfinite parameter state")
    atomic_json(args.metadata_dir / "metrics.json", metrics)
    if args.probe_batches:
        atomic_json(args.metadata_dir / "smoke-complete.json", metrics)
    else:
        atomic_json(args.metadata_dir / ".complete.json", {"configuration": configuration, "job": job.__dict__})
    return 0
