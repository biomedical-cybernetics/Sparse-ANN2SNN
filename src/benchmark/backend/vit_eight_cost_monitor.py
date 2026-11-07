"""Monitor GPU memory with NVML."""
from __future__ import annotations

import ctypes as c
import json
import os
import time
from pathlib import Path


class Memory(c.Structure):
    # v2 distinguishes firmware/driver-reserved memory from active device usage.
    _fields_ = [("version", c.c_uint), ("total", c.c_ulonglong), ("reserved", c.c_ulonglong),
                ("free", c.c_ulonglong), ("used", c.c_ulonglong)]


class Process(c.Structure):
    _fields_ = [("pid", c.c_uint), ("usedGpuMemory", c.c_ulonglong),
                ("gpuInstanceId", c.c_uint), ("computeInstanceId", c.c_uint)]


class Util(c.Structure):
    _fields_ = [("gpu", c.c_uint), ("memory", c.c_uint)]


class Nvml:
    def __init__(self, gpu_ids):
        self.lib = c.CDLL("libnvidia-ml.so.1")
        self._check(self.lib.nvmlInit_v2())
        self.handles = {}
        self.metadata = {}
        self.telemetry = {}
        self.last_telemetry = 0.
        for gpu in gpu_ids:
            handle = c.c_void_p()
            self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(c.c_uint(int(gpu)), c.byref(handle)))
            self.handles[str(gpu)] = handle
            info = {}
            for key, name in (("name", "nvmlDeviceGetName"), ("uuid", "nvmlDeviceGetUUID")):
                buf = c.create_string_buffer(128)
                self._check(getattr(self.lib, name)(handle, buf, c.c_uint(len(buf))))
                info[key] = buf.value.decode()
            self.metadata[str(gpu)] = info

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"NVML call failed: code={code}; ownership cannot be assumed")

    def close(self):
        self._check(self.lib.nvmlShutdown())

    def pids(self):
        result = set()
        for handle in self.handles.values():
            for name in ("nvmlDeviceGetComputeRunningProcesses_v2", "nvmlDeviceGetGraphicsRunningProcesses_v2"):
                function = getattr(self.lib, name)
                for _ in range(5):
                    count = c.c_uint(128)
                    array = (Process * count.value)()
                    code = function(handle, c.byref(count), array)
                    if code == 7:
                        # Retry with a fresh capacity; reject unbounded churn.
                        if count.value > 128:
                            raise RuntimeError("NVML process count exceeds safe ownership-query capacity")
                        continue
                    self._check(code)
                    result.update(int(array[i].pid) for i in range(count.value))
                    break
                else:
                    raise RuntimeError("NVML process list did not stabilize")
        return result

    def sample(self):
        start = time.monotonic_ns()
        cards = []
        for gpu, handle in self.handles.items():
            memory, util = Memory(), Util()
            memory.version = c.sizeof(Memory) | (2 << 24)
            self._check(self.lib.nvmlDeviceGetMemoryInfo_v2(handle, c.byref(memory)))
            cards.append(dict(gpu=gpu, **self.metadata[gpu], sampled_ns=time.monotonic_ns(), used_bytes=int(memory.used),
                reserved_bytes=int(memory.reserved), total_bytes=int(memory.total)))
        end = time.monotonic_ns()
        # Keep the eight memory queries close together. Slower telemetry is 1Hz.
        refresh = time.monotonic()-self.last_telemetry >= 1.
        for card in cards:
            gpu, handle = card["gpu"], self.handles[card["gpu"]]
            if not refresh:
                card.update(self.telemetry[gpu])
                continue
            util = Util()
            self._check(self.lib.nvmlDeviceGetUtilizationRates(handle, c.byref(util)))
            fields = {}
            for key, name, extra in (
                ("power_mw", "nvmlDeviceGetPowerUsage", ()),
                ("temperature_c", "nvmlDeviceGetTemperature", (c.c_uint(0),)),
                ("sm_clock_mhz", "nvmlDeviceGetClockInfo", (c.c_uint(1),)),
                ("memory_clock_mhz", "nvmlDeviceGetClockInfo", (c.c_uint(2),)),
            ):
                value = c.c_uint()
                code = getattr(self.lib, name)(handle, *extra, c.byref(value))
                # Optional telemetry may be unsupported, unlike ownership queries.
                fields[key] = int(value.value) if code == 0 else None
            self.telemetry[gpu] = dict(utilization=int(util.gpu), telemetry_ns=time.monotonic_ns(), **fields)
            card.update(self.telemetry[gpu])
        if refresh:
            self.last_telemetry = time.monotonic()
        return dict(start_ns=start, end_ns=end, telemetry_end_ns=time.monotonic_ns(), cards=cards, memory_api="nvmlMemory_v2",
            aggregate_used_bytes=sum(x["used_bytes"] for x in cards))


def descendant(pid, root):
    """Elastic ranks can start their own sessions: ancestry, not PGID alone, owns them."""
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == root:
            return True
        seen.add(pid)
        try:
            lines = Path(f"/proc/{pid}/status").read_text().splitlines()
            pid = int(next(x.split()[1] for x in lines if x.startswith("PPid:")))
        except (OSError, StopIteration, ValueError) as exc:
            raise RuntimeError("Cannot verify GPU process ancestry") from exc
    return False


def descendant_groups(root):
    groups = {os.getpgid(root)}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if descendant(pid, root):
                groups.add(os.getpgid(pid))
        except (RuntimeError, ProcessLookupError):
            # A racing unrelated/terminated PID is not admitted to cleanup.
            continue
    groups.discard(os.getpgrp())
    return groups


class GpuPidVanished(RuntimeError):
    def __init__(self, pid):
        self.pid = pid
        super().__init__(f"GPU PID vanished during ownership lookup: {pid}")


def pid_lineage(pid):
    """Failure-only process identity evidence; exclude command-line arguments."""
    rows = []
    for _ in range(32):
        try:
            raw = Path(f"/proc/{pid}/stat").read_text()
            fields = raw[raw.rfind(")")+2:].split()
            rows.append(dict(pid=pid, comm=raw[raw.find("(")+1:raw.rfind(")")],
                state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]),
                session=int(fields[3]), start_ticks=int(fields[19])))
        except (OSError, ValueError, IndexError) as exc:
            rows.append(dict(pid=pid, error=type(exc).__name__))
            break
        parent = int(fields[1])
        if parent <= 0 or parent == pid:
            break
        pid = parent
    return rows


def require_ownership(pids, allowed_group=None):
    foreign = []
    for pid in pids:
        try:
            group = os.getpgid(pid)
        except ProcessLookupError as exc:
            raise GpuPidVanished(pid) from exc
        except PermissionError as exc:
            raise RuntimeError(f"Cannot establish GPU PID ownership: {pid}") from exc
        if allowed_group is None or (group != allowed_group and not descendant(pid, allowed_group)):
            foreign.append(pid)
    if foreign:
        evidence = {pid: pid_lineage(pid) for pid in foreign}
        raise RuntimeError(f"Exclusive GPUs violated by non-experiment PIDs: {sorted(foreign)}; "
            f"allowed_root={allowed_group}; lineage={json.dumps(evidence, sort_keys=True)}")


def ownership_round(nvml, root, verified):
    """Recheck ownership when a previously verified process exits during sampling."""
    pending = None
    vanished = []
    for _ in range(4):
        started = time.monotonic_ns()
        pids = nvml.pids() if pending is None else pending
        retry = False
        for pid in sorted(pids):
            try:
                require_ownership({pid}, root)
            except GpuPidVanished:
                fresh = nvml.pids()
                if pid not in verified or pid in fresh or Path(f"/proc/{pid}").exists():
                    raise
                vanished.append(pid)
                pending, retry = fresh, True
                break
            else:
                verified.add(pid)
        if not retry:
            return started, sorted(pids), vanished
    raise RuntimeError("GPU PID list did not stabilize during verified-exit recheck")


def peak_for_window(samples, start_ns, end_ns):
    # All eight sequential queries must fall inside the measurement window.
    inside = [s for s in samples if s["start_ns"] >= start_ns and s["end_ns"] <= end_ns]
    if not inside:
        raise RuntimeError("No complete NVML sampling round in measurement window")
    best = max(inside, key=lambda s: s["aggregate_used_bytes"])
    return dict(aggregate_peak_gib=best["aggregate_used_bytes"] / 2**30,
        peak_sample=best, sample_rounds=len(inside),
        maximum_sampling_span_ms=max(s["end_ns"]-s["start_ns"] for s in inside)/1e6)


def monitor_process(process, nvml, path, interval=.1, ownership_interval=1., on_tick=None, exclusive=True):
    """Sample GPU memory; the caller handles process cleanup on errors."""
    samples = []
    next_owner = 0.
    last_safe_ns = 0
    observed_pids = []
    verified = set()
    with path.open("w", buffering=1) as handle:
        while process.poll() is None:
            started = time.monotonic()
            if started >= next_owner:
                if exclusive:
                    owner_round_started_ns, observed_pids, vanished = ownership_round(nvml, process.pid, verified)
                else:
                    owner_round_started_ns = time.monotonic_ns()
                    observed_pids, vanished = sorted(nvml.pids()), []
                last_safe_ns = owner_round_started_ns
                next_owner = started + ownership_interval
            sample = nvml.sample()
            sample["last_ownership_round_start_ns"] = last_safe_ns
            sample["exclusive_guard_enforced"] = exclusive
            sample["observed_gpu_pids"] = observed_pids
            sample["verified_exited_pids"] = vanished
            samples.append(sample)
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            if on_tick:
                on_tick(samples, last_safe_ns)
            time.sleep(max(0., interval - (time.monotonic()-started)))
    if exclusive:
        owner_round_started_ns, observed_pids, vanished = ownership_round(nvml, process.pid, verified)
    else:
        owner_round_started_ns = time.monotonic_ns()
    if on_tick:
        on_tick(samples, owner_round_started_ns)
    return samples
