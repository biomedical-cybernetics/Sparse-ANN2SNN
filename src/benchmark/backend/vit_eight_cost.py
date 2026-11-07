"""DeiT training-cost settings and paired measurement units."""
from __future__ import annotations
import hashlib
import json
import math
from statistics import mean, stdev
EXPERIMENT_ID = "deit-training-cost"
SEEDS = (0, 1, 2)
PHASES = {"pre": (10, 6), "active": (180, 101), "post": (110, 246)}
MODES = ("pipeline", "compute")
LIMITS = []


def parse_gpu_lane(value):
    items=value.split(",")
    if len(items)!=8 or any(not x.isdigit() for x in items):
        raise ValueError("Require exactly eight distinct comma-separated physical GPU IDs")
    ids=[int(x) for x in items]
    if len(set(ids))!=8:
        raise ValueError("Physical GPU IDs must be distinct, including zero-prefixed aliases")
    return [str(x) for x in ids]

def amp_step_summary(attempted, successful, scale_before, scale_after):
    if (type(attempted) is not int or type(successful) is not int or
            not 0 <= successful <= attempted or attempted < 1):
        raise ValueError("Invalid optimizer step accounting")
    if not all(math.isfinite(v) and v > 0 for v in (scale_before, scale_after)):
        raise ValueError("Invalid AMP scale")
    return dict(attempted=attempted, successful=successful, skipped=attempted-successful,
        scale_before=scale_before, scale_after=scale_after)

def config():
    return dict(schema_version=1, experiment_id=EXPERIMENT_ID, seeds=list(SEEDS), world_size=8,
        model="deit_base_patch16_224", input_size=224, classes=1000, dataset="ImageNet-1K",
        batch=128, global_batch=1024, workers=10, epochs=300, steps_per_epoch=1251,
        amp="fp16", scaler=True, parameters="fp32", optimizer_states="fp32", tf32=False,
        amp_step_policy="native-dynamic-scaler-record-skips-v2",
        cudnn_benchmark=True, strict_determinism=False, optimizer="adamw", base_lr=5e-4,
        lr=.001, weight_decay=.05, opt_eps=1e-8, betas=[.9, .999], min_lr=1e-5,
        warmup_lr=1e-6, warmup_epochs=dict(dense=5, cht=10), drop=0., drop_path=.1,
        ema_decay=.99996, mixup=.8, cutmix=1., smoothing=.1, color_jitter=.3,
        aa="rand-m9-mstd0.5-inc1", interpolation="bicubic", reprob=.25,
        remode="pixel", recount=1, repeated_aug=3, eval_crop_ratio=.875,
        warmup_steps=50, ordinary_rounds=4, ordinary_batches=100, update_repeats=3,
        update_batches=10, update_batch=5, profile_batches=10, nvml_interval_s=.1,
        ownership_interval_s=1., val_batch=192, val_samples=50000,
        phases={key: dict(epochs=v[0], logical_epoch=v[1]) for key, v in PHASES.items()},
        cht=dict(sparsity=.7, BRF=True, brf_r=.1, degree_dist="uniform",
            remove_method="weight_magnitude_soft", regrow_method="CH2_L3n_soft",
            adaptive_zeta=True, zeta=.1, start_T=1., end_T=9., update_interval=5,
            dst_start_epoch=10, dst_end_epoch_fraction=.6, sparse_tensors=72,
            chain_removal=False, granet=False), limits=LIMITS)

def update_epochs():
    # The production scheduler advances only after the first ten epochs.
    return list(range(15, 191, 5))

def counter_before(epoch, steps=1251):
    return max(0, epoch - 1 - 10) * steps

def units():
    """One unit is the smallest recoverable complete pair (or auxiliary pass)."""
    result = []
    for mode in MODES:
        for seed in SEEDS:
            for phase in PHASES:
                for repeat in range(4):
                    result.append(dict(id=f"{mode}-s{seed}-{phase}-r{repeat}", kind="ordinary",
                        mode=mode, seed=seed, phase=phase, repeat=repeat,
                        members=["dense", "cht"] if repeat % 2 == 0 else ["cht", "dense"]))
            for j, epoch in enumerate(update_epochs(), 1):
                for repeat in range(3):
                    result.append(dict(id=f"{mode}-s{seed}-u{j:02d}-r{repeat}", kind="update",
                        mode=mode, seed=seed, progress=j, logical_epoch=epoch, repeat=repeat,
                        members=["without", "with"] if repeat % 2 == 0 else ["with", "without"]))
    for seed in SEEDS:
        for phase in PHASES:
            for method in ("dense", "cht"):
                result.append(dict(id=f"profile-s{seed}-{phase}-{method}", kind="profile",
                    mode="profile", seed=seed, phase=phase, method=method))
        for j, epoch in enumerate(update_epochs(), 1):
            result.append(dict(id=f"profile-s{seed}-u{j:02d}", kind="profile-update",
                mode="profile", seed=seed, progress=j, logical_epoch=epoch, method="cht"))
        for method in ("dense", "cht"):
            result.append(dict(id=f"aux-s{seed}-{method}", kind="aux", mode="aux",
                seed=seed, method=method))
    return result

def data_seed(unit, rank):
    # No method, branch, run ID, worker assignment, or output path enters pairing.
    # Pipeline and compute replay the same inputs as well.
    key = [unit.get(k) for k in ("kind","seed","phase","progress","repeat")] if "kind" in unit else [unit["id"]]
    encoded = json.dumps([key, rank, "input"], sort_keys=True, separators=(",", ":")).encode()
    return int(hashlib.sha256(encoded).hexdigest()[:8], 16)
