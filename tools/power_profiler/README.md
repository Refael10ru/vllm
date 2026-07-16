# vLLM GPU power/perf sweep

Offline profiler: sweeps (SM clock x mem clock) grid, runs prefill-only and
decode-only vLLM workloads at each point, samples power draw. Output feeds a
runtime DVFS policy (lower SM clock during decode-dominant regimes).

## Setup (on the GPU rig)

```bash
uv pip install pynvml matplotlib
# passwordless sudo for nvidia-smi only:
echo "$USER ALL=(root) NOPASSWD: $(which nvidia-smi)" | sudo tee /etc/sudoers.d/nvidia-smi
```

## Run

```bash
# ~30-60 min: 6 SM x 4 mem points, 30s per phase per point, crash-safe resume
python power_sweep.py --model meta-llama/Llama-3.1-8B-Instruct

# clock-switch settle/stall microbenchmark (decides per-batch vs windowed policy)
python power_sweep.py --measure-switch-latency

python plot_sweep.py results.json   # -> heatmaps.png, pareto.png
```

Notes:

- Mem clock locking (`-lmc`) is unreliable on consumer Ampere; the sweep
  auto-degrades to SM-axis-only if it fails.
- 30-series power sensor reports board power at ~10Hz; per-point averages over
  30s are trustworthy, sub-second numbers are not.
- Clocks are reset (`-rgc`/`-rmc`) on exit, including Ctrl-C.
