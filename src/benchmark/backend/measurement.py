"""Paired timing, memory sampling and result validation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from runtime import atomic_json
from vit_eight_cost import config, amp_step_summary
from vit_eight_cost_monitor import monitor_process, peak_for_window, require_ownership, descendant_groups

def available_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def shared_smoke(manifest):
    allowed = manifest.get("smoke_allow_occupied", False)
    if allowed and not manifest.get("smoke", False):
        raise ValueError("smoke_allow_occupied requires smoke=True")
    return allowed

def check_access(nvml, manifest):
    pids = sorted(nvml.pids())
    if not shared_smoke(manifest):
        require_ownership(pids)
    return pids

def validate_result(result, unit, identity, require_amp=True):
    if result.get("identity") != identity or result.get("unit") != unit:
        raise RuntimeError("Unit identity mismatch")
    if unit["kind"] == "aux":
        if result["auxiliary"]["validation"]["samples"] != 50000:
            raise RuntimeError("Incomplete validation")
        return
    members = result["members"]
    keys = unit.get("members", [unit.get("method")])
    if sorted(members) != sorted(keys):
        raise RuntimeError("An interrupted pair cannot be reused")
    for value in members.values():
        if value["elapsed_s"] <= 0 or len(value["ranks"]) != 8:
            raise RuntimeError("Incomplete eight-rank timing")
        if value.get("sample_rounds", 0) < 1:
            raise RuntimeError("Measurement lacks NVML coverage")
        if require_amp:
            expected = 2 if result.get("smoke", False) else (100 if unit["kind"] == "ordinary" else 10)
            amp = value["amp"]
            if amp != amp_step_summary(expected, amp["successful"], amp["scale_before"], amp["scale_after"]):
                raise RuntimeError("Invalid AMP measurement accounting")
            if any(rank.get("amp") != amp for rank in value["ranks"]):
                raise RuntimeError("Missing or inconsistent eight-rank AMP accounting")
        if "dataset_identity" in result and any(r.get("dataset_identity") != result["dataset_identity"] for r in value["ranks"]):
            raise RuntimeError("Dataset identity differs between ranks")
    if len(keys) == 2:
        a, b = [members[k] for k in keys]
        for ra, rb in zip(a["ranks"], b["ranks"]):
            if (ra["rank"], ra["indices"], ra["input_shapes"]) != (
                    rb["rank"], rb["indices"], rb["input_shapes"]):
                raise RuntimeError("Paired sample order or input shape mismatch")
    if unit["kind"] == "update":
        if members["with"]["topology_updates"] != unit["progress"] or members["without"]["topology_updates"] != unit["progress"]:
            raise RuntimeError("Topology replay did not reach the expected counter")

def stop_group(process):
    if process.poll() is not None:
        return
    groups = descendant_groups(process.pid)
    for sig, timeout in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            for group in groups:
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                alive = []
                for group in groups:
                    try:
                        os.killpg(group, 0)
                        alive.append(group)
                    except ProcessLookupError:
                        pass
                if not alive:
                    return
                time.sleep(.1)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue

def execute_group(manifest, work, member, group_name, nvml):
    raw = Path(manifest["raw"])
    group = raw/"workers"/group_name
    group.mkdir(parents=True, exist_ok=False)
    output = Path(manifest["checkpoint"])/group_name
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(group/"units.json", work)
    initial_pids = check_access(nvml, manifest)
    cmd = [manifest["python"], "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=8",
        "--rdzv-backend=c10d", f"--rdzv-endpoint=localhost:{available_port()}",
        str(raw/"source"/"vit_eight_cost_worker.py"), "--manifest", str(raw/"manifest.json"),
        "--unit-file", str(group/"units.json"), "--member", member, "--output", str(output)]
    env = dict(os.environ)
    if manifest.get("nccl_runtime"):
        runtime = manifest["nccl_runtime"]
        if not Path(runtime["library"]).is_file():
            raise RuntimeError("Selected NCCL library is missing")
        env["LD_PRELOAD"] = runtime["library"]+" "+env.get("LD_PRELOAD", "")
    env.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(manifest["gpu_ids"]),
        NVIDIA_TF32_OVERRIDE="0", PYTHONPATH=str(raw/"source"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    atomic_json(group/"launch.json", dict(command=cmd, environment={k:env.get(k) for k in (
        "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "OMP_NUM_THREADS")},
        nccl_environment={k:v for k,v in env.items() if k.startswith("NCCL_") or k == "LD_PRELOAD"},
        initial_gpu_pids=initial_pids, exclusive_guard_enforced=not shared_smoke(manifest),
        formal_measurement_valid=not manifest["smoke"]))
    pending = {u["id"]:u for u in work}
    def decorate(value, samples):
        for measured in value.get("members", {}).values():
            measured.update(peak_for_window(samples, measured["start_ns"], measured["end_ns"]))
        value["stage_memory"] = []
        for interval in value.get("intervals", []):
            try:
                value["stage_memory"].append(dict(interval, **peak_for_window(samples, interval["start_ns"], interval["end_ns"])))
            except RuntimeError:
                value["stage_memory"].append(dict(interval, status="BelowSamplingResolution"))
        value["worker_evidence"] = str(group)
        return value
    def commit_ready(samples, safe_ns):
        # Publish complete topology pairs while the bundle is still running.
        # Wait for a successful ownership check AFTER the pair finished.
        for name, unit in list(pending.items()):
            if unit["kind"] != "update":
                continue
            p = output/(name+".json")
            if not p.exists():
                continue
            value = json.loads(p.read_text())
            if max(v["end_ns"] for v in value["members"].values()) > safe_ns:
                continue
            publish(manifest, decorate(value, samples))
            del pending[name]
    with (group/"worker.log").open("w") as log:
        process = subprocess.Popen(cmd, env=env, cwd=raw/"source", stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        samples = []
        try:
            samples = monitor_process(process, nvml, group/"nvml.jsonl", on_tick=commit_ready,
                exclusive=not shared_smoke(manifest))
            code = process.wait()
            if code:
                raise RuntimeError(f"DDP worker failed: exit={code}; see {group/'worker.log'}")
        except BaseException:
            stop_group(process)
            atomic_json(group/"failed.json", dict(error=str(sys.exc_info()[1]), timestamp_ns=time.monotonic_ns(),
                uncommitted_pairs_invalid=True, previously_committed_pairs_retained=True))
            raise
    outputs = {}
    for unit in work:
        path = output/(unit["id"]+".json")
        value = json.loads(path.read_text())
        outputs[unit["id"]] = decorate(value, samples)
    return outputs

def publish(manifest, result):
    result["smoke"] = manifest.get("smoke", False)
    result["smoke_allow_occupied"] = shared_smoke(manifest)
    result["formal_measurement_valid"] = not result["smoke"]
    validate_result(result, result["unit"], manifest["identity"])
    data_identity = result.get("dataset_identity")
    if not data_identity:
        raise RuntimeError("Missing dataset identity")
    frozen = Path(manifest["raw"])/"input-identity.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != data_identity:
            raise RuntimeError("Dataset listing/classes changed between measurements or recovery")
    else:
        atomic_json(frozen, data_identity)
    path = Path(manifest["raw"])/"completed"/(result["unit"]["id"]+".json")
    if path.exists():
        raise RuntimeError("Refusing duplicate successful unit")
    atomic_json(path, result)

def completed(manifest):
    path = Path(manifest["raw"])/"completed"
    found = []
    for unit in manifest["units"]:
        p = path/(unit["id"]+".json")
        if p.exists():
            record = json.loads(p.read_text())
            validate_result(record, unit, manifest["identity"], require_amp=
                manifest.get("config",config()).get("amp_step_policy") == "native-dynamic-scaler-record-skips-v2")
            found.append(record)
    return found
