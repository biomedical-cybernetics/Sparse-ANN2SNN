from __future__ import annotations
from typing import Any,Callable
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

def _is_distributed(args: Any) -> bool:
    return bool(
        args is not None
        and getattr(args, "distributed", False)
        and dist.is_available()
        and dist.is_initialized()
    )

def make_dcgs_evaluator(
    *,
    monitor_type: type,
    reset: Callable[[Any], None],
    capture: dict[str, np.ndarray],
    max_batches: int | None,
    diagnostic: dict[str, Any] | None = None,
    collect_operations: bool = True,
) -> Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return a signature-compatible DCGS SOP evaluator."""

    def evaluate(model, test_loader, device, args=None, **kwargs):
        real_model = model.module if hasattr(model, "module") else model
        timesteps = int(real_model.T)
        num_activations = int(kwargs.get("num_activations", 0))
        mac_layers = set(kwargs.get("MACLayers") or [])
        monitor = (
            monitor_type(
                model,
                step_mode=real_model.step_mode,
                T=timesteps,
                coding_type=real_model.coding_type,
            )
            if collect_operations
            else None
        )
        model.eval()
        collector = None
        ann_reference = None
        collect_density = "imagenet" not in str(
            getattr(args, "dataset", "")
        ).lower()

        correct = np.zeros(timesteps, dtype=np.float64)
        total = np.zeros(timesteps, dtype=np.float64)
        mac = np.zeros(timesteps, dtype=np.float64)
        sop = np.zeros(timesteps, dtype=np.float64)
        spikes = np.zeros((num_activations, timesteps), dtype=np.float64)
        layer_names = list(monitor.monitored_layers) if monitor is not None else []
        layer_mac = np.zeros((len(layer_names), timesteps), dtype=np.float64)
        layer_sop = np.zeros((len(layer_names), timesteps), dtype=np.float64)
        layer_input_events = np.zeros(
            (len(layer_names), timesteps), dtype=np.float64
        )
        length = 0.0
        sampler_indices = list(iter(test_loader.sampler))
        sample_cursor = 0

        iterator = test_loader
        if not _is_distributed(args) or dist.get_rank() == 0:
            iterator = tqdm(test_loader, desc="ANP-SOP recount")

        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(iterator):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch_indices = sampler_indices[
                    sample_cursor : sample_cursor + int(targets.size(0))
                ]
                sample_cursor += int(targets.size(0))
                rank = dist.get_rank() if _is_distributed(args) else 0
                sample_ids = [
                    f"rank-{rank}:dataset-index-{int(index)}"
                    for index in batch_indices
                ]
                reset(model)
                inputs = inputs.to(device)
                outputs = model(inputs)
                length += float(targets.size(0))

                for timestep in range(timesteps):
                    predicted = outputs[: timestep + 1].mean(0).cpu().max(1)[1]
                    correct[timestep] += float(predicted.eq(targets).sum().item())
                    total[timestep] += float(targets.size(0))

                if monitor is not None:
                    layer_index = 0
                    density_available = False
                    for monitored_index, name in enumerate(monitor.monitored_layers):
                        records = monitor[name]
                        if "diff" in real_model.coding_type:
                            records = records[1:]
                        if records:
                            record_has_density = len(records[0]) > 2
                            density_available = (
                                density_available or record_has_density
                            )
                            for timestep in range(timesteps):
                                operations = float(records[timestep][0])
                                if name in mac_layers:
                                    mac[timestep] += operations
                                    layer_mac[monitored_index, timestep] += operations
                                else:
                                    sop[timestep] += operations
                                    layer_sop[monitored_index, timestep] += operations
                                if record_has_density:
                                    layer_input_events[
                                        monitored_index, timestep
                                    ] += float(records[timestep][2])
                                if (
                                    collect_density
                                    and record_has_density
                                    and layer_index > 0
                                    and "shortcut" not in name
                                    and "downsample" not in name
                                    and layer_index - 1 < num_activations
                                ):
                                    spikes[layer_index - 1, timestep] += float(
                                        records[timestep][2]
                                    )
                            if (
                                collect_density
                                and record_has_density
                                and "shortcut" not in name
                                and "downsample" not in name
                            ):
                                layer_index += 1
                    if num_activations and density_available and collect_density:
                        assert layer_index - 1 == num_activations, (
                            f"collected {layer_index - 1} activations, "
                            f"expected {num_activations}"
                        )
                    monitor.clear_recorded_data()

        if _is_distributed(args):
            packed = torch.tensor(
                np.concatenate(
                    [correct, total, mac, sop, spikes.reshape(-1), [length]]
                ),
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            values = packed.cpu().numpy()
            offset = 0
            correct = values[offset : offset + timesteps]
            offset += timesteps
            total = values[offset : offset + timesteps]
            offset += timesteps
            mac = values[offset : offset + timesteps]
            offset += timesteps
            sop = values[offset : offset + timesteps]
            offset += timesteps
            spike_count = num_activations * timesteps
            spikes = values[offset : offset + spike_count].reshape(
                num_activations, timesteps
            )
            offset += spike_count
            length = float(values[offset])
            if layer_names:
                operation_pack = torch.tensor(
                    np.concatenate(
                        [
                            layer_mac.reshape(-1),
                            layer_sop.reshape(-1),
                            layer_input_events.reshape(-1),
                        ]
                    ),
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(operation_pack, op=dist.ReduceOp.SUM)
                operation_values = operation_pack.cpu().numpy()
                layer_size = len(layer_names) * timesteps
                layer_mac = operation_values[:layer_size].reshape(
                    len(layer_names), timesteps
                )
                layer_sop = operation_values[
                    layer_size : 2 * layer_size
                ].reshape(len(layer_names), timesteps)
                layer_input_events = operation_values[
                    2 * layer_size : 3 * layer_size
                ].reshape(len(layer_names), timesteps)

        if length <= 0 or np.any(total <= 0):
            raise RuntimeError("ANP-SOP recount evaluated no samples")
        mac_ops = np.cumsum(mac) / length
        sop_ops = np.cumsum(sop) / length
        capture["samples"] = length
        capture["MACOps"] = mac_ops
        capture["SOPOps"] = sop_ops
        capture["TotalOps"] = sop_ops.copy()
        capture["LayerNames"] = np.asarray(layer_names, dtype=str)
        capture["LayerMACOps"] = np.cumsum(layer_mac, axis=1) / length
        capture["LayerSOPOps"] = np.cumsum(layer_sop, axis=1) / length
        capture["LayerInputEventRate"] = layer_input_events / length
        capture["LayerAverageFiringRate"] = (
            np.cumsum(layer_input_events, axis=1)
            / np.arange(1, timesteps + 1)
            / length
        )
        legacy_energy = (np.cumsum(mac) * 4.6 + np.cumsum(sop) * 0.9) / length * 1e-6
        lasfr = np.zeros_like(spikes)
        for layer in range(num_activations):
            lasfr[layer] = np.cumsum(spikes[layer]) / np.arange(1, timesteps + 1)
        lasfr /= length
        return correct / total, lasfr, legacy_energy

    return evaluate
