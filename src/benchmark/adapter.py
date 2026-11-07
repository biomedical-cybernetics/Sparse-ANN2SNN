"""Standalone entry for the paired eight-GPU DeiT cost benchmark."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import signal
import sys
from statistics import mean

BACKEND = Path(__file__).resolve().parent / "backend"


def project_seed(records, seed):
    """Project measured step times to 300 epochs; report the simultaneous eight-GPU memory peak."""
    by_id = {record["unit"]["id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate measurement units")
    result = {"seed": seed, "epochs": 300, "methods": {}}
    for method in ("dense", "cht"):
        values = {}
        for mode in ("pipeline", "compute"):
            total = 0.0
            for phase, epochs in (("pre", 10), ("active", 180), ("post", 110)):
                names = [f"{mode}-s{seed}-{phase}-r{repeat}" for repeat in range(4)]
                total += 1251 * epochs * mean(by_id[name]["members"][method]["elapsed_s"] / 100 for name in names)
            if method == "cht":
                for position in range(1, 37):
                    names = [f"{mode}-s{seed}-u{position:02d}-r{repeat}" for repeat in range(3)]
                    total += mean(by_id[name]["members"]["with"]["elapsed_s"] -
                                  by_id[name]["members"]["without"]["elapsed_s"] for name in names)
            values[f"projected_{mode}_hours"] = total / 3600
        peaks = [member["aggregate_peak_gib"] for record in records
                 if record["unit"]["seed"] == seed and record["unit"]["mode"] == "pipeline"
                 and record["unit"]["kind"] in ("ordinary", "update")
                 for name, member in record["members"].items()
                 if (name == method if record["unit"]["kind"] == "ordinary" else method == "cht" and name == "with")]
        values["aggregate_peak_gpu_memory_gib"] = max(peaks)
        result["methods"][method] = values
    return result


def execute(recipe, runtime):
    if runtime.get("resume"):
        raise ValueError("Use a new output directory for each complete benchmark seed")
    sys.path.insert(0, str(BACKEND))
    settings = importlib.import_module("vit_eight_cost")
    measurement = importlib.import_module("measurement")
    monitor = importlib.import_module("vit_eight_cost_monitor")
    atomic_json = importlib.import_module("runtime").atomic_json
    gpu_ids = settings.parse_gpu_lane(runtime.get("gpus") or os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    output = Path(runtime["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir()
    for item in BACKEND.glob("*.py"):
        shutil.copyfile(item, source / item.name)
    nccl = None
    if runtime.get("nccl_library"):
        library = Path(runtime["nccl_library"]).resolve(strict=True)
        nccl = {"library": str(library)}
    work = [unit for unit in settings.units() if unit["seed"] == runtime["seed"]]
    configuration = settings.config()
    identity = dict(configuration=configuration, nccl_library=nccl["library"] if nccl else None)
    manifest = dict(identity=identity, config=configuration, units=work,
                    gpu_ids=gpu_ids, python=sys.executable, data_root=runtime["data_root"],
                    raw=str(output), checkpoint=str(output / "states"),
                    smoke=False, smoke_allow_occupied=False, nccl_runtime=nccl)
    atomic_json(output / "manifest.json", manifest)
    nvml = monitor.Nvml(gpu_ids)
    previous = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Benchmark interrupted by signal {signum}")
    try:
        measurement.check_access(nvml, manifest)
        atomic_json(output / "baseline.json", nvml.sample())
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupted)
        consumed = set()
        for unit in work:
            if unit["id"] in consumed:
                continue
            if unit["kind"] == "ordinary":
                result = dict(unit=unit, identity=identity, members={}, worker_evidence=[])
                for member in unit["members"]:
                    value = measurement.execute_group(manifest, [unit], member, unit["id"] + "-" + member, nvml)[unit["id"]]
                    result["members"].update(value["members"])
                    result["worker_evidence"].append(value["worker_evidence"])
                    result.setdefault("stage_memory", []).extend(value["stage_memory"])
                    if "dataset_identity" in result and result["dataset_identity"] != value["dataset_identity"]:
                        raise RuntimeError("Dataset changed between paired methods")
                    result["dataset_identity"] = value["dataset_identity"]
                measurement.publish(manifest, result)
                consumed.add(unit["id"])
            else:
                group = [u for u in work if u["kind"] == unit["kind"] and u["seed"] == unit["seed"]
                         and u["mode"] == unit["mode"] and u["id"] not in consumed]
                if unit["kind"] != "update":
                    group = [unit]
                values = measurement.execute_group(manifest, group, "", unit["id"] + "-bundle", nvml)
                for value in values.values():
                    if not (output / "completed" / (value["unit"]["id"] + ".json")).exists():
                        measurement.publish(manifest, value)
                    consumed.add(value["unit"]["id"])
        records = measurement.completed(manifest)
        if len(records) != len(work):
            raise RuntimeError("Incomplete benchmark measurements")
        atomic_json(output / "measurement.json", project_seed(records, runtime["seed"]))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        nvml.close()
