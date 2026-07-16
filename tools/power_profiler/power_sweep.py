#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline (SM clock x mem clock) power/perf sweep for vLLM on NVIDIA GPUs.

For each clock grid point, runs a prefill-only and a decode-only workload,
sampling power draw. Writes rows to results.json (crash-safe, incremental):

    {sm_mhz, mem_mhz, phase, tok_s, watts_mean, watts_p95, joules, j_per_tok}

Clock setting shells out to `sudo nvidia-smi` (needs passwordless sudo for
nvidia-smi only). Measurement is unprivileged pynvml. Clocks are reset on exit.

Usage:
    python power_sweep.py --model meta-llama/Llama-3.1-8B-Instruct
    python power_sweep.py --measure-switch-latency
"""

import argparse
import atexit
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import pynvml


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


class ClockControl:
    def __init__(self, gpu: int):
        self.gpu = str(gpu)
        atexit.register(self.reset)

    def _smi(self, *args: str) -> bool:
        r = sh(["sudo", "nvidia-smi", "-i", self.gpu, *args])
        if r.returncode != 0:
            print(f"nvidia-smi {args} failed: {r.stderr.strip()}")
        return r.returncode == 0

    def set_sm(self, mhz: int) -> bool:
        return self._smi("-lgc", f"{mhz},{mhz}")

    def set_mem(self, mhz: int) -> bool:
        return self._smi("-lmc", f"{mhz},{mhz}")

    def reset(self):
        self._smi("-rgc")
        self._smi("-rmc")


class PowerSampler:
    """Samples power at ~10Hz on a thread; prefers the HW energy counter."""

    def __init__(self, handle, interval: float = 0.1):
        self.handle = handle
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._energy_start: int | None = None

    def _energy_mj(self) -> int | None:
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except pynvml.NVMLError:
            return None

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
            self._stop.wait(self.interval)

    def start(self):
        self.samples = []
        self._energy_start = self._energy_mj()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = time.perf_counter()
        self._thread.start()

    def stop(self) -> dict:
        elapsed = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join()
        energy_end = self._energy_mj()
        if self._energy_start is not None and energy_end is not None:
            joules = (energy_end - self._energy_start) / 1000.0
            energy_source = "hw_counter"
        else:
            joules = statistics.fmean(self.samples) * elapsed if self.samples else 0.0
            energy_source = "power_integral"
        return {
            "watts_mean": round(statistics.fmean(self.samples), 1),
            "watts_p95": round(
                statistics.quantiles(self.samples, n=20)[-1], 1
            )
            if len(self.samples) >= 20
            else max(self.samples),
            "joules": round(joules, 1),
            "elapsed_s": round(elapsed, 2),
            "energy_source": energy_source,
        }


def pick_grid(handle, sm_points: int, mem_points: int) -> tuple[list[int], list[int]]:
    """Evenly-spaced subsets of the supported clock steps (descending)."""
    mem_clocks = sorted(pynvml.nvmlDeviceGetSupportedMemoryClocks(handle), reverse=True)
    sm_clocks = sorted(
        pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, mem_clocks[0]), reverse=True
    )

    def subsample(xs: list[int], n: int) -> list[int]:
        if len(xs) <= n:
            return xs
        step = (len(xs) - 1) / (n - 1)
        return [xs[round(i * step)] for i in range(n)]

    return subsample(sm_clocks, sm_points), subsample(mem_clocks, mem_points)


def build_workloads(llm, prefill_len: int, decode_prompt_len: int, decode_len: int):
    import random

    from vllm import SamplingParams

    rng = random.Random(0)
    tok_ids = lambda n: [rng.randint(1000, 20000) for _ in range(n)]
    prefill_prompts = [{"prompt_token_ids": tok_ids(prefill_len)} for _ in range(16)]
    decode_prompts = [{"prompt_token_ids": tok_ids(decode_prompt_len)} for _ in range(64)]
    prefill_sp = SamplingParams(max_tokens=1, ignore_eos=True)
    decode_sp = SamplingParams(max_tokens=decode_len, ignore_eos=True)

    def run_prefill() -> int:
        llm.generate(prefill_prompts, prefill_sp, use_tqdm=False)
        return 16 * prefill_len

    def run_decode() -> int:
        llm.generate(decode_prompts, decode_sp, use_tqdm=False)
        return 64 * decode_len

    return {"prefill": run_prefill, "decode": run_decode}


def measure_phase(run_fn, sampler: PowerSampler, min_duration: float) -> dict:
    sampler.start()
    tokens = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < min_duration:
        tokens += run_fn()
    elapsed = time.perf_counter() - t0
    stats = sampler.stop()
    tok_s = tokens / elapsed
    stats.update(
        tok_s=round(tok_s, 1),
        j_per_tok=round(stats["joules"] / tokens, 4) if tokens else None,
        tokens=tokens,
    )
    return stats


def measure_switch_latency(handle, clocks: ClockControl, sm_clocks: list[int]) -> None:
    """Settle time of -lgc while GPU is busy, and whether the load stalls."""
    import torch

    a = torch.randn(4096, 4096, device="cuda")
    gaps: list[float] = []
    stop = threading.Event()

    def busy():
        nonlocal gaps
        last = time.perf_counter()
        while not stop.is_set():
            (a @ a).sum().item()
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    t = threading.Thread(target=busy, daemon=True)
    t.start()
    time.sleep(2)
    for target in [sm_clocks[-1], sm_clocks[0]]:
        baseline = statistics.fmean(gaps[-20:])
        t0 = time.perf_counter()
        clocks.set_sm(target)
        while time.perf_counter() - t0 < 2.0:
            if abs(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM) - target) < 30:
                break
            time.sleep(0.001)
        settle_ms = (time.perf_counter() - t0) * 1000
        time.sleep(0.5)
        stall_ms = max(gaps[-int(0.5 / baseline) :]) * 1000
        print(
            f"switch to {target}MHz: settle {settle_ms:.1f}ms, "
            f"max iter gap {stall_ms:.1f}ms (baseline {baseline * 1000:.1f}ms)"
        )
    stop.set()
    t.join()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--sm-points", type=int, default=6)
    p.add_argument("--mem-points", type=int, default=4)
    p.add_argument("--duration", type=float, default=30.0, help="seconds per phase")
    p.add_argument("--prefill-len", type=int, default=2048)
    p.add_argument("--decode-prompt-len", type=int, default=32)
    p.add_argument("--decode-len", type=int, default=256)
    p.add_argument("--output", default="results.json")
    p.add_argument("--measure-switch-latency", action="store_true")
    args = p.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    clocks = ClockControl(args.gpu)
    sm_clocks, mem_clocks = pick_grid(handle, args.sm_points, args.mem_points)
    print(f"GPU: {gpu_name}\nSM grid: {sm_clocks}\nMem grid: {mem_clocks}")

    if args.measure_switch_latency:
        measure_switch_latency(handle, clocks, sm_clocks)
        return

    if not clocks.set_mem(mem_clocks[0]):
        print("mem clock locking unsupported; sweeping SM axis only")
        mem_clocks = [pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)]
        mem_lock = False
    else:
        mem_lock = True

    from vllm import LLM

    llm = LLM(model=args.model, max_model_len=args.prefill_len + 8)
    workloads = build_workloads(
        llm, args.prefill_len, args.decode_prompt_len, args.decode_len
    )
    sampler = PowerSampler(handle)

    out = Path(args.output)
    results: list[dict] = json.loads(out.read_text()) if out.exists() else []
    done = {(r["sm_mhz"], r["mem_mhz"], r["phase"]) for r in results}

    for mem in mem_clocks:
        if mem_lock and not clocks.set_mem(mem):
            continue
        for sm in sm_clocks:
            if not clocks.set_sm(sm):
                continue
            time.sleep(1.0)
            for phase, run_fn in workloads.items():
                if (sm, mem, phase) in done:
                    continue
                run_fn()  # warmup at this clock
                stats = measure_phase(run_fn, sampler, args.duration)
                row = {
                    "gpu": gpu_name,
                    "model": args.model,
                    "sm_mhz": sm,
                    "mem_mhz": mem,
                    "phase": phase,
                    **stats,
                }
                print(row)
                results.append(row)
                out.write_text(json.dumps(results, indent=1))

    clocks.reset()
    print(f"done: {len(results)} rows -> {out}")


if __name__ == "__main__":
    main()
