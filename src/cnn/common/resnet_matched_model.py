import copy
import math
import torch
from torch import nn
from Models.ResNet import resnet18


def model_for(job):
    return resnet18(num_classes=job.classes, in_channels=3)


def weight_count(model):
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))


def target_budget(job):
    with torch.random.fork_rng(devices=[]): model = model_for(job)
    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)): continue
        n = m.weight.numel()
        sparse = isinstance(m, nn.Conv2d) and "shortcut" not in name
        pairs = m.in_channels * m.out_channels if sparse else n
        active = (pairs - math.floor(.7 * pairs)) * math.prod(m.kernel_size) if sparse else n
        rows.append({"layer": name, "shape": list(m.weight.shape), "weights": n,
                     "active_weights": active, "mask_sparsity": 1 - active / n,
                     "granularity": "channel-pair" if sparse else "dense"})
    return {"layers": rows, "weights": sum(r["weights"] for r in rows),
            "active_weights": sum(r["active_weights"] for r in rows)}


def scheduler_for(model, epochs, lr, warmup=5):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9,
                                weight_decay=5e-4, nesterov=True)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup, eta_min=0.)
    if not warmup: return optimizer, cosine
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.1, end_factor=1., total_iters=warmup)
    return optimizer, torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cosine], milestones=[warmup])


def prune_trial(source, example, ratio):
    import torch_pruning as tp
    model = copy.deepcopy(source).eval()
    head = [m for m in model.modules() if isinstance(m, nn.Linear)]
    pruner = tp.pruner.GroupNormPruner(model, example,
        importance=tp.importance.GroupMagnitudeImportance(p=2), iterative_steps=1,
        global_pruning=True, pruning_ratio=ratio, max_pruning_ratio=1., ignored_layers=head)
    pruner.step()
    with torch.no_grad():
        out = model(example)
    if out.shape != (example.shape[0], head[0].out_features): raise ValueError("Invalid pruned output")
    return model


def calibrate_budget(source, example, target, tolerance=.01, trial=prune_trial):
    """Independent single-shot trials; no accuracy evaluation or cumulative pruning."""
    points = []
    best = None
    def evaluate(ratio):
        nonlocal best
        try:
            candidate = trial(source, example, ratio)
            n = weight_count(candidate)
            row = {"ratio": float(ratio), "weights": n, "valid": True}
            key = (abs(n-target), -n, float(ratio))
            if best is None or key < best[0]: best = (key, float(ratio), n)
        except (ValueError, RuntimeError, IndexError, AssertionError) as error:
            if "out of memory" in str(error).lower(): raise
            row = {"ratio": float(ratio), "valid": False, "error": repr(error)}
        points.append(row)
        return row
    for i in range(64): evaluate(.99*i/63)
    intervals = [(a,b) for a,b in zip(points, points[1:]) if a["valid"] and b["valid"] and (a["weights"]-target)*(b["weights"]-target) <= 0]
    if intervals:
        a,b = min(intervals, key=lambda pair: pair[0]["ratio"])
        for _ in range(32):
            mid = evaluate((a["ratio"]+b["ratio"])/2)
            if not mid["valid"]: break
            if (a["weights"]-target)*(mid["weights"]-target) <= 0: b=mid
            else: a=mid
    if best is None or best[0][0]/target > tolerance:
        error=ValueError(f"No legal DepGraph budget within {tolerance}; target={target}, best={best}")
        error.audit={"target_weights":target,"trials":points,"best_count_only_candidate":best}
        raise error
    model = trial(source, example, best[1])
    return model, {"target_weights": target, "actual_weights": weight_count(model),
                   "relative_error": abs(weight_count(model)-target)/target,
                   "ratio": best[1], "trials": points, "selection": "count-only"}
