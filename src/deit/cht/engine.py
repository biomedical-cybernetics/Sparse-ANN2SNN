# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from precision import autocast_context


def _tensor_failure_summary(tensor):
    if not torch.is_tensor(tensor):
        return {"type": type(tensor).__name__}
    detached = tensor.detach()
    summary = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
    }
    if torch.is_floating_point(detached) or torch.is_complex(detached):
        finite = torch.isfinite(detached)
        summary.update({
            "elements": detached.numel(),
            "finite": int(finite.sum().item()),
            "nan": int(torch.isnan(detached).sum().item()),
            "inf": int(torch.isinf(detached).sum().item()),
        })
        if finite.any().item():
            finite_values = detached[finite]
            summary["finite_min"] = float(finite_values.min().item())
            summary["finite_max"] = float(finite_values.max().item())
    return summary


def _record_nonfinite_loss(model, samples, targets, outputs, loss_value, epoch, batch_index, args):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    bad_parameters = []
    for name, parameter in model.named_parameters():
        if (torch.is_floating_point(parameter) or torch.is_complex(parameter)) and (
                not torch.isfinite(parameter).all().item()):
            bad_parameters.append(name)
    record = {
        "event": "nonfinite_loss",
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", rank)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "epoch": epoch,
        "batch_index": batch_index,
        "loss": str(loss_value),
        "samples": _tensor_failure_summary(samples),
        "targets": _tensor_failure_summary(targets),
        "outputs": _tensor_failure_summary(outputs),
        "nonfinite_parameter_names": bad_parameters,
    }
    message = "[NONFINITE_DIAGNOSTIC] " + json.dumps(record, sort_keys=True)
    print(message, file=sys.stderr, flush=True)
    if getattr(args, "output_dir", None):
        path = Path(args.output_dir) / (
            f"nonfinite-rank-{rank}-epoch-{epoch}-batch-{batch_index}.json"
        )
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def train_one_epoch(model: torch.nn.Module, pruner, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, iterative_warmup_steps: int,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model = model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    topology_update_seconds = 0.0
    topology_update_count = 0
    topology_memory: list[dict[str, float | int]] = []

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    for batch_index, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.probe_batches is not None and batch_index >= args.probe_batches:
            break
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with autocast_context(args.amp_dtype):
            #assert model.module.distilled_training, 'must enable distilled training'
            outputs = model(samples)

            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets)
                loss = loss + 0.25 * criterion(outputs[1], targets)
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            _record_nonfinite_loss(
                model, samples, targets, outputs, loss_value,
                epoch, batch_index, args,
            )
            print("Loss is {}, stopping training".format(loss_value), flush=True)
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        if hasattr(lr_scheduler, "iterative_step"):
            lr_scheduler.iterative_step() #do iterative warm up here. will auto finish.

        if pruner:
            measure_update = args.timing_components and pruner.will_update_next_call()
            if measure_update:
                torch.cuda.synchronize()
                topology_started = time.perf_counter()
            no_dst = pruner()
            if measure_update:
                torch.cuda.synchronize()
                topology_update_seconds += time.perf_counter() - topology_started
                topology_update_count += 1
                topology_memory.append({
                    "rank": utils.get_rank(),
                    "allocated_gib": round(torch.cuda.memory_allocated(device) / 1024**3, 4),
                    "reserved_gib": round(torch.cuda.memory_reserved(device) / 1024**3, 4),
                    "max_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 4),
                    "max_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 1024**3, 4),
                })
            if not no_dst and iterative_warmup_steps > 0 and hasattr(lr_scheduler, "start_iterative_warmup"):
                print(f"Start iterative warmup for {iterative_warmup_steps} steps!")
                lr_scheduler.start_iterative_warmup(iterative_warmup_steps)
                #after dst, activate iterative warm up in scheduler.

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.timing_components:
        timing = torch.tensor(
            [topology_update_seconds, float(topology_update_count), -float(topology_update_count)],
            device=device,
            dtype=torch.float64,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(timing, op=torch.distributed.ReduceOp.MAX)
        maximum_count = int(timing[1].item())
        minimum_count = int(-timing[2].item())
        if minimum_count != maximum_count:
            raise RuntimeError(
                f"topology update count differs across ranks: min={minimum_count}, max={maximum_count}"
            )
        result["topology_update_seconds"] = float(timing[0].item())
        result["topology_update_count"] = maximum_count
        if torch.distributed.is_initialized():
            gathered = [None for _ in range(utils.get_world_size())]
            torch.distributed.all_gather_object(gathered, topology_memory)
            result["topology_memory"] = [item[0] for item in gathered if item]
        else:
            result["topology_memory"] = topology_memory
    return result


@torch.no_grad()
def evaluate(data_loader, model, device, amp_dtype="fp16", max_batches=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch_index, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with autocast_context(amp_dtype):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
