from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import random
import time

# Set before importing torch: the server environment may force TF32 on.
os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, get_state_dict

import models  # noqa: F401; the same local model registration as full training
import utils
from datasets import build_dataset
from dst_scheduler import DSTScheduler
from dst_util import recur_split_atten
from precision import autocast_context, build_loss_scaler
from runtime import atomic_json
from samplers import RASampler
from vit_eight_cost import PHASES, config, counter_before, data_seed, amp_step_summary


class OptimizerStepCounter:
    """CPU-only post-hook; counts actual updates, including through the DST wrapper."""
    def __init__(self):
        self.total = 0

    def __call__(self, optimizer, args, kwargs):
        self.total += 1


def tensors_finite(tensors):
    # Adam's counters may be CPU tensors even when its moments live on CUDA.
    by_device = {}
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            by_device.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    return all(bool(torch.stack(values).all().item()) for values in by_device.values())


def rng():
    # Do not initialize CUDA contexts on the other seven devices.
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def set_rng(state):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def restore_optimizer_in_place(optimizer, saved):
    if len(optimizer.param_groups)!=len(saved["param_groups"]):
        raise RuntimeError("Warmed optimizer group layout changed")
    for current_group,saved_group in zip(optimizer.param_groups,saved["param_groups"]):
        if len(current_group["params"])!=len(saved_group["params"]):
            raise RuntimeError("Warmed optimizer parameter layout changed")
        for parameter,index in zip(current_group["params"],saved_group["params"]):
            source=saved["state"].get(index,{})
            target=optimizer.state.get(parameter,{})
            if set(source)!=set(target):
                raise RuntimeError("Warmed optimizer state keys changed")
            for key,value in source.items():
                if torch.is_tensor(value):
                    if not torch.is_tensor(target[key]) or target[key].shape!=value.shape or target[key].dtype!=value.dtype:
                        raise RuntimeError("Warmed optimizer tensor layout changed")
                    target[key].copy_(value)
                else:
                    target[key]=copy.deepcopy(value)
        for key,value in saved_group.items():
            if key!="params":
                current_group[key]=copy.deepcopy(value)


class Indexed(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index


class ValidationShard(Sampler):
    def __init__(self, size, rank, world):
        self.indices = range(rank, size, world)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def shutdown(loader):
    """Release one rank-owned iterator, including pinned persistent-worker handles.

    This worker deliberately owns at most one active DataLoader at a time. PyTorch
    registers process objects at exit when persistence and pinning are both on;
    stopping workers alone leaves their sentinel/pipe handles retained by atexit.
    All of this cleanup runs after the measured block, without consuming RNG.
    """
    import atexit
    import multiprocessing

    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        workers = tuple(iterator._workers)
        iterator._shutdown_workers()
        for worker in workers:
            worker.join(timeout=5.)
            if worker.is_alive():
                raise RuntimeError("DataLoader worker did not exit during bounded cleanup")
        # unregister removes all registrations of this callback, including those
        # for other loaders. Other child processes need their exit callbacks.
        if multiprocessing.active_children():
            raise RuntimeError("Cannot release DataLoader exit callbacks with other active children")
        if iterator._persistent_workers and iterator._pin_memory:
            atexit.unregister(type(iterator)._clean_up_worker)
        for worker in workers:
            worker.close()
        loader._iterator = None


class Probe:
    def __init__(self, manifest, method, seed, output):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.method, self.seed, self.output = method, seed, output
        self.identity = manifest["identity"]
        self.checkpoint_root = Path(manifest["checkpoint"])
        self.c = config()
        self.profiles = {}
        self.profile = False
        self.call_pruner = False
        self.smoke = manifest.get("smoke", False)
        self.intervals = []
        started = time.monotonic_ns()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        a = argparse.Namespace(data_set="IMNET", data_path=manifest["data_root"], input_size=224,
            color_jitter=.3, aa=self.c["aa"], train_interpolation="bicubic", reprob=.25,
            remode="pixel", recount=1, resplit=False, ThreeAugment=False, src=False,
            inat_category="name", eval_crop_ratio=.875)
        dataset, classes = build_dataset(True, a)
        self.dataset, self.data_args = Indexed(dataset), a
        self.sampler = RASampler(self.dataset, num_replicas=8, rank=self.rank, shuffle=True)
        if classes != 1000 or len(dataset.classes) != 1000 or len(dataset) != 1281167 or len(self.sampler)//128 != 1251:
            raise RuntimeError("ImageNet identity/loader contract is not 1281167/1000/1251")
        self.dataset_identity = dict(train_root=str(Path(a.data_path)/"train"),
            classes=dataset.class_to_idx, train_samples=len(dataset), validation_samples=50000)
        a.opt, a.lr, a.weight_decay, a.opt_eps, a.opt_betas = "adamw", .001, .05, 1e-8, None
        a.momentum = .9
        a.sched, a.epochs, a.min_lr = "cosine", 300, 1e-5
        a.warmup_epochs, a.warmup_lr = (10 if method == "cht" else 5), 1e-6
        a.cooldown_epochs, a.patience_epochs, a.decay_epochs, a.decay_rate = 10, 10, 30, .1
        a.lr_noise, a.lr_noise_pct, a.lr_noise_std, a.seed = None, .67, 1., seed
        a.amp_dtype, a.model_ema, a.model_ema_decay, a.model_ema_force_cpu = "fp16", True, .99996, False
        for key, value in self.c["cht"].items():
            setattr(a, key, value)
        for key, value in dict(sparsity=.7 if method == "cht" else 0., sparsity_distribution="uniform",
                mask=None, WS=False, ws_beta=.25, EM_S=False, itop=False, early_stop=False,
                gmp=False, granet=False, pruning_T_end=None, history_weights=False,
                new_history_weights=False, ssam=False).items():
            setattr(a, key, value)
        self.args = a
        self.base = create_model("deit_base_patch16_224", pretrained=False, num_classes=1000,
            drop_rate=0., drop_path_rate=.1).to(self.device)
        if method == "cht":
            recur_split_atten(self.base)
        self.ema = ModelEma(self.base, decay=.99996, device="", resume="")
        ddp_started = time.monotonic_ns()
        if self.rank == 0:
            print("[cost-probe] Entering eight-rank DDP setup", flush=True)
        self.model = torch.nn.parallel.DistributedDataParallel(self.base, device_ids=[self.device.index])
        torch.cuda.synchronize()
        if self.rank == 0:
            print("[cost-probe] Eight-rank DDP setup completed", flush=True)
        self.intervals.append(dict(stage="ddp_setup", start_ns=ddp_started, end_ns=time.monotonic_ns()))
        self.optimizer = create_optimizer(a, self.base)
        self.scheduler, _ = create_scheduler(a, self.optimizer)
        self.scaler = build_loss_scaler("fp16")
        self.pruner = None
        if method == "cht":
            self.pruner = DSTScheduler(self.model, self.optimizer, alpha=.1, delta=5*1251,
                T_end=180*1251, ignore_linear_layers=False, sparsity_distribution="uniform",
                grad_accumulation_n=1, args=a)
            if len(self.pruner.W) != 72:
                raise RuntimeError("Expected 72 sparse projection/MLP tensors")
        self.mixup = Mixup(mixup_alpha=.8, cutmix_alpha=1., prob=1., switch_prob=.5,
            mode="batch", label_smoothing=.1, num_classes=1000)
        self.criterion = SoftTargetCrossEntropy()
        self.optimizer_step_counter = OptimizerStepCounter()
        self.optimizer_step_hook = self.optimizer.register_step_post_hook(self.optimizer_step_counter)
        torch.cuda.synchronize()
        self.intervals.append(dict(stage="initialization", start_ns=started, end_ns=time.monotonic_ns()))
        warm_unit = dict(id=f"warm-s{seed}")
        loader = self.loader(warm_unit)
        self.measure(loader, 2 if self.smoke else 50, "pipeline", False, "warmup", warm_unit)
        shutdown(loader)
        if self.rank == 0:
            print("[cost-probe] Production-batch warmup completed", flush=True)
        self.optimizer.zero_grad(set_to_none=True)
        cache = self.checkpoint_root/"warm"/f"s{seed}-{method}.pth"
        exists = [cache.exists() if self.rank == 0 else None]
        dist.broadcast_object_list(exists,src=0)
        if not exists[0]:
            common = dict(model=cpu_copy(self.base.state_dict()), optimizer=cpu_copy(self.optimizer.state_dict()),
                scheduler=cpu_copy(self.scheduler.state_dict()), ema=cpu_copy(get_state_dict(self.ema)),
                scaler=cpu_copy(self.scaler.state_dict()), pruner=cpu_copy(self.pruner.state_dict()) if self.pruner else None)
            rngs = [None]*8
            dist.all_gather_object(rngs,cpu_copy(rng()))
            if self.rank == 0:
                cache.parent.mkdir(parents=True,exist_ok=True)
                temporary = cache.with_suffix(".tmp")
                torch.save(dict(identity=self.identity,dataset_identity=self.dataset_identity,
                    state=common,rng_by_rank=rngs),temporary)
                os.replace(temporary,cache)
            del common
        dist.barrier()
        fixed = torch.load(cache,map_location="cpu",weights_only=False)
        if fixed["identity"] != self.identity or fixed["dataset_identity"] != self.dataset_identity:
            raise RuntimeError("Fixed warm snapshot identity mismatch")
        self.warm = fixed["state"]
        self.warm["rng"] = fixed["rng_by_rank"][self.rank]

    def loader(self, unit):
        value = data_seed(unit, self.rank)
        self.sampler.set_epoch(value)
        gen = torch.Generator().manual_seed(value)
        return DataLoader(self.dataset, sampler=self.sampler, batch_size=128,
            num_workers=10, pin_memory=True, drop_last=True, persistent_workers=True, generator=gen)

    def restore(self, unit):
        self.base.load_state_dict(self.warm["model"])
        restore_optimizer_in_place(self.optimizer,self.warm["optimizer"])
        self.scheduler.load_state_dict(self.warm["scheduler"])
        utils._load_checkpoint_for_ema(self.ema, self.warm["ema"])
        self.scaler.load_state_dict(self.warm["scaler"])
        if self.pruner:
            self.pruner.load_state_dict(self.warm["pruner"])
            for hook in self.pruner.backward_hook_objects:
                if hook:
                    hook.dense_grad = None
            epoch = unit.get("logical_epoch", PHASES.get(unit.get("phase"), (0, 101))[1])
            self.pruner.step = counter_before(epoch)
            if "progress" in unit:
                # Batch five is the real update call at every progress point.
                self.pruner.step = unit["progress"] * 5*1251 - 5
                if self.smoke:
                    self.pruner.step += 4
            self.pruner.dst_steps = max(0, self.pruner.step // (5*1251))
        epoch = unit.get("logical_epoch", PHASES.get(unit.get("phase"), (0, 101))[1])
        # Production steps the scheduler at the END of zero-based epoch.
        self.scheduler.step(max(0, epoch - 2))
        self.call_pruner = epoch > 10
        set_rng(self.warm["rng"])
        # Identical model-training streams across the two branches of a pair.
        value = data_seed(unit, self.rank)
        torch.manual_seed(value)
        np.random.seed(value)
        random.seed(value)
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        self.model.train()
        self.profiles = {}

    @contextmanager
    def section(self, name):
        if not self.profile:
            yield
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.profiles.setdefault(name, []).append(time.perf_counter()-start)

    def install_profile_wrappers(self):
        if not self.pruner:
            return
        # Wrappers synchronize only in the independent profiling pass.
        for name in ("link_removal", "link_regrowth", "_scores_for_regrowth", "reset_momentum",
                     "apply_mask_to_weights", "apply_mask_to_gradients"):
            original = getattr(self.pruner, name)
            def wrapped(*args, _name=name, _original=original, **kwargs):
                with self.section("topology." + _name):
                    return _original(*args, **kwargs)
            setattr(self.pruner, name, wrapped)
        original_broadcast = dist.broadcast
        def broadcast(*args, **kwargs):
            with self.section("topology.broadcast_overlaps_regrowth"):
                return original_broadcast(*args, **kwargs)
        dist.broadcast = broadcast

    def train(self, x, y, update_allowed=True, warm=False):
        with self.section("mixup"):
            x, y = self.mixup(x, y)
        with self.section("forward_loss"):
            with autocast_context("fp16"):
                output = self.model(x)
                if output.dtype != torch.float16:
                    raise RuntimeError("Autocast forward did not produce FP16 logits")
                loss = self.criterion(output, y)
            value = loss.item()  # The production loop also materializes loss here.
            if not math.isfinite(value):
                raise RuntimeError("Non-finite training loss; scientific fallback is prohibited")
        self.optimizer.zero_grad()
        if self.profile:
            # Identical NativeScaler operations, split only for independent profiling.
            with self.section("backward_ddp_and_gradient_mask"):
                self.scaler._scaler.scale(loss).backward(create_graph=False)
            with self.section("optimizer_scaler_and_state_weight_masks"):
                self.scaler._scaler.step(self.optimizer)
                self.scaler._scaler.update()
        else:
            self.scaler(loss, self.optimizer, parameters=self.model.parameters())
        if self.pruner and self.call_pruner:
            next_update = self.pruner.will_update_next_call()
            with self.section("pruner_update" if next_update else "pruner_bookkeeping"):
                if next_update:
                    saved = rng()
                    original = self.pruner._dst_step
                    try:
                        if not update_allowed:
                            self.pruner._dst_step = lambda: None
                        self.pruner()
                    finally:
                        self.pruner._dst_step = original
                        set_rng(saved)
                else:
                    self.pruner()
        # This synchronization precedes EMA in the production engine.
        torch.cuda.synchronize()
        with self.section("ema"):
            self.ema.update(self.model)
        return value

    def measure(self, loader, batches, mode, update_allowed, stage, unit):
        iterator_started = time.monotonic_ns()
        iterator = iter(loader)
        # Exclude iterator/cold first-batch startup, not ongoing pipeline waiting.
        # This is a steady-state projection, never an entire epoch wall-clock.
        first_batch = next(iterator)
        self.intervals.append(dict(stage="iterator_and_first_batch_startup", start_ns=iterator_started, end_ns=time.monotonic_ns()))
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        initial_scale = self.scaler.state_dict()["scale"]
        initial_successful_steps = self.optimizer_step_counter.total
        start = time.monotonic_ns()
        elapsed = 0.
        indices = []
        retained = []
        proof = []
        for i in range(batches):
            with self.section("data_wait_cpu_augmentation"):
                x, y, index = first_batch if i == 0 else next(iterator)
            indices.extend(index.tolist())
            if i in (0, batches-1):
                # Inspect retained batch shapes after the timed region.
                retained.append((x, y))
            with self.section("h2d"):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            if mode == "compute":
                torch.cuda.synchronize()
                compute_started = time.perf_counter()
            self.train(x, y, update_allowed, warm=stage == "warmup")
            if mode == "compute":
                torch.cuda.synchronize()
                elapsed += time.perf_counter()-compute_started
        torch.cuda.synchronize()
        end = time.monotonic_ns()
        if mode != "compute":
            elapsed = (end-start)/1e9
        # Read scaler/counter state OUTSIDE primary timing. A native AMP overflow
        # is not an invalid model update: GradScaler correctly skips that update.
        amp = amp_step_summary(batches, self.optimizer_step_counter.total-initial_successful_steps,
            initial_scale, self.scaler.state_dict()["scale"])
        for x, y in retained:
            proof.append([list(x.shape), list(y.shape)])
        local = dict(elapsed_s=elapsed, start_ns=start, end_ns=end, amp=amp,
            dataset_identity=self.dataset_identity,
            device_uuid=str(getattr(torch.cuda.get_device_properties(self.device), "uuid", "unavailable")),
            indices=indices, input_shapes=proof, rank=self.rank,
            allocated_peak_gib=torch.cuda.max_memory_allocated()/2**30,
            reserved_peak_gib=torch.cuda.max_memory_reserved()/2**30,
            profile=self.profiles, lr=self.optimizer.param_groups[0]["lr"],
            scaler=self.scaler.state_dict(), sparse_tensors=len(self.pruner.W) if self.pruner else 0,
            mask_counts=[int(m.sum()) for m in self.pruner.backward_masks] if self.pruner else [])
        ranks = [None] * 8
        dist.all_gather_object(ranks, local)
        if any(r["amp"] != amp for r in ranks):
            raise RuntimeError("AMP optimizer step/scale accounting differs across DDP ranks")
        self.check_precision()
        result = dict(elapsed_s=max(r["elapsed_s"] for r in ranks),
            amp=amp,
            start_ns=min(r["start_ns"] for r in ranks), end_ns=max(r["end_ns"] for r in ranks),
            ranks=ranks, method=self.method, stage=stage, unit=unit["id"], mode=mode,
            topology_step=self.pruner.step if self.pruner else None,
            topology_updates=self.pruner.dst_steps if self.pruner else 0)
        self.intervals.append(dict(stage=stage, start_ns=result["start_ns"], end_ns=result["end_ns"]))
        return result

    def check_precision(self):
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError("TF32 was re-enabled")
        for p in self.base.parameters():
            if p.dtype != torch.float32 or (p.grad is not None and p.grad.dtype != torch.float32):
                raise RuntimeError("Parameter/gradient is not FP32")
        for state in self.optimizer.state.values():
            if any(torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
                raise RuntimeError("Optimizer floating state is not FP32")
        if not tensors_finite(self.base.parameters()):
            raise RuntimeError("Non-finite model parameter; AMP skip is not a fallback for corrupted weights")
        if not tensors_finite(v for state in self.optimizer.state.values() for v in state.values()):
            raise RuntimeError("Non-finite optimizer state")

    def auxiliary(self, unit):
        self.restore(unit)
        dataset, _ = build_dataset(False, self.data_args)
        if len(dataset) != 50000:
            raise RuntimeError("Validation must contain exactly 50000 images")
        if dataset.class_to_idx != self.dataset.dataset.class_to_idx:
            raise RuntimeError("Training/validation class identities differ")
        loader = DataLoader(Indexed(dataset), sampler=ValidationShard(len(dataset), self.rank, 8),
            batch_size=192, num_workers=10, pin_memory=True, drop_last=False)
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        totals = torch.zeros(4, device=self.device, dtype=torch.float64)
        torch.cuda.synchronize()
        dist.barrier()
        start = time.monotonic_ns()
        seen = []
        with torch.no_grad():
            for x, y, ids in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with autocast_context("fp16"):
                    output = self.model(x)
                    loss = criterion(output, y)
                predicted = output.topk(5, dim=1).indices
                totals += torch.tensor([loss.item()*len(y),
                    (predicted[:, 0] == y).sum().item(),
                    (predicted == y[:, None]).any(1).sum().item(), len(y)], device=self.device)
                seen.extend(ids.tolist())
        dist.all_reduce(totals)
        torch.cuda.synchronize()
        end = time.monotonic_ns()
        seen_all = [None] * 8
        dist.all_gather_object(seen_all, seen)
        if sorted(i for seq in seen_all for i in seq) != list(range(50000)):
            raise RuntimeError("Validation samples were duplicated or omitted")
        elapsed = torch.tensor((end-start)/1e9,device=self.device,dtype=torch.float64)
        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        validation = dict(stage="validation", start_ns=start, end_ns=end, elapsed_s=float(elapsed), samples=50000,
            per_rank_batch=192, accuracy_not_reported=True)
        self.intervals.append(validation)
        states = [None]*8
        dist.all_gather_object(states, rng())
        io = {}
        if self.rank == 0:
            start = time.monotonic_ns()
            payload = dict(model=self.base.state_dict(), optimizer=self.optimizer.state_dict(),
                lr_scheduler=self.scheduler.state_dict(), epoch=100, model_ema=get_state_dict(self.ema),
                scaler=self.scaler.state_dict(), args=self.args, rng_states_by_rank=states,
                benchmark_only=True, identity=self.identity)
            io["state_preparation_s"] = (time.monotonic_ns()-start)/1e9
            for name, value in [("checkpoint.pth", payload)] + (
                    [("pruner.pth", self.pruner.state_dict())] if self.pruner else []):
                path = self.output/name
                started = time.monotonic_ns()
                torch.save(value, path)
                written = time.monotonic_ns()
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                durable = time.monotonic_ns()
                loaded = torch.load(path, map_location="cpu", weights_only=False)
                if name == "checkpoint.pth" and loaded["identity"] != self.identity:
                    raise RuntimeError("Checkpoint readback identity mismatch")
                if name == "checkpoint.pth":
                    if any(not torch.equal(loaded["model"][key], value.detach().cpu())
                           for key, value in self.base.state_dict().items()):
                        raise RuntimeError("Checkpoint model weights failed exact readback")
                if name == "pruner.pth" and loaded["step"] != self.pruner.step:
                    raise RuntimeError("Pruner readback mismatch")
                if name == "pruner.pth" and any(not torch.equal(a,b.cpu()) for a,b in
                        zip(loaded["backward_masks"], self.pruner.backward_masks)):
                    raise RuntimeError("Pruner masks failed exact readback")
                io[name] = dict(write_return_s=(written-started)/1e9, fsync_extra_s=(durable-written)/1e9,
                    bytes=path.stat().st_size, start_ns=started, end_ns=durable, readback=True)
        all_io = [io]
        dist.broadcast_object_list(all_io, src=0)
        self.intervals.append(dict(stage="checkpoint", start_ns=min(
            v["start_ns"] for v in all_io[0].values() if isinstance(v, dict)),
            end_ns=max(v["end_ns"] for v in all_io[0].values() if isinstance(v, dict))))
        return dict(validation=validation, checkpoint=all_io[0], benchmark_only=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--unit-file", required=True)
    parser.add_argument("--member", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    nccl = manifest.get("nccl_runtime")
    identity = dict(configuration=config(), nccl_library=nccl["library"] if nccl else None)
    if config() != manifest["config"] or identity != manifest["identity"]:
        raise RuntimeError("Benchmark configuration mismatch")
    if manifest.get("nccl_runtime"):
        runtime = manifest["nccl_runtime"]
        libraries = {s.split()[-1] for s in Path("/proc/self/maps").read_text().splitlines() if "libnccl.so" in s}
        if runtime["library"] not in libraries:
            raise RuntimeError("Explicit NCCL runtime was not loaded")
    work = json.loads(Path(args.unit_file).read_text())
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    dist.init_process_group("nccl")
    if dist.get_world_size() != 8:
        raise RuntimeError("This experiment requires exactly eight ranks")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    first = work[0]
    method = first.get("method", "cht" if first["kind"] != "ordinary" else args.member)
    probe = Probe(manifest, method, first["seed"], output)
    if first["kind"].startswith("profile"):
        probe.install_profile_wrappers()
    for unit in work:
        probe.restore(unit)
        if unit["kind"] == "aux":
            result = dict(unit=unit, auxiliary=probe.auxiliary(unit))
        else:
            probe.profile = unit["kind"].startswith("profile")
            members = unit["members"] if unit["kind"] == "update" else [args.member or method]
            measured = {}
            for member in members:
                probe.restore(unit)
                loader = probe.loader(unit)
                batches = 100 if unit["kind"] == "ordinary" else 10
                if manifest.get("smoke"):
                    batches = 2
                measured[member] = probe.measure(loader, batches, unit["mode"],
                    member != "without", "profile" if probe.profile else "training", unit)
                shutdown(loader)
                expected = unit.get("progress", 0)
                if unit["kind"] in ("update", "profile-update"):
                    if probe.pruner.dst_steps != expected:
                        raise RuntimeError("Update counter mismatch")
            result = dict(unit=unit, members=measured)
        result.update(identity=manifest["identity"], dataset_identity=probe.dataset_identity,
            intervals=list(probe.intervals), precision=dict(
            amp="fp16", scaler=True, tf32=False, parameters="fp32", optimizer_states="fp32"))
        import timm
        result["environment"] = dict(torch=torch.__version__, cuda=torch.version.cuda,
            nccl_version=torch.cuda.nccl.version(),
            nccl_libraries=sorted({s.split()[-1] for s in Path("/proc/self/maps").read_text().splitlines() if "libnccl.so" in s}),
            cudnn=torch.backends.cudnn.version(), timm=timm.__version__, numpy=np.__version__,
            torch_cpu_threads=torch.get_num_threads(), cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            tf32_override=os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
            nvidia_tf32_override=os.environ.get("NVIDIA_TF32_OVERRIDE"))
        if dist.get_rank() == 0:
            atomic_json(output/(unit["id"]+".json"), result)
        probe.intervals.clear()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
