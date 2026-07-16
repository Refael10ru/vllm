# vLLM DVFS power-optimization plan

Goal: cut GPU power in vLLM serving by lowering clocks where the workload
doesn't need them, with per-(model, GPU) profiles measured offline.

## Decisions (grill session 2026-07-16)

| Decision | Answer |
|---|---|
| Objective | Max perf/watt under latency SLO (~<=5% TTFT/TBT regression) |
| Runtime target | Online serving, mixed continuous batching (chunked prefill) |
| Rig | Consumer RTX 30-series (Ampere), 7-8B model, passwordless `sudo nvidia-smi` |
| Actuation knob | SM clock lock only (`-lgc`); mem lock unreliable on consumer Ampere |
| Switching policy | Per-batch threshold if clock switch doesn't stall the GPU, windowed regime detector (2-5s window + hysteresis) if it does — decided by measured switch latency |
| Granularity | Phase-level (prefill=compute-bound, decode=memory-bound), NOT per-kernel: clock switch ~10-50ms settle, device-wide; kernels are us-ms |
| Profile storage | JSON keyed (model, gpu, phase, sm_mhz, mem_mhz) — not vLLM dataclasses. IrOpImpl hertz-fields idea dropped (branch `worktree-iropimpl-hertz-fields` obsolete) |
| vLLM config profiled | Production (torch.compile + cudagraphs); no vLLM core changes needed |

## Architecture

1. **Offline profiler** (`power_sweep.py`, DONE, untested on GPU):
   sweep (SM x mem) clock grid; per point run prefill-only (16x2048-tok
   prompts, max_tokens=1) and decode-only (64 prompts x 256 gen toks,
   ignore_eos) workloads; sample power 10Hz via pynvml (HW energy counter
   when available); crash-safe incremental results.json; clocks reset on
   exit. `--measure-switch-latency` microbench: settle time + stall gap
   under busy matmul — picks the switching policy.
2. **Plots** (`plot_sweep.py`, DONE): heatmaps tok/s, watts, tok/s/W over
   (sm_hz x mem_hz) per phase; perf-vs-power pareto with SM-MHz labels.
3. **Runtime applier** (NOT BUILT): loads profile JSON; hooks
   `GPUModelRunner.execute_model` (vllm/v1/worker/gpu_model_runner.py,
   `_model_forward` ~:3811) or scheduler stats for prefill/decode token
   ratio; sets SM clock via root helper when regime shifts. v2.

## Key facts learned

- Clock changes: device-wide, need root, ~10-50ms settle; DVFS transitions
  do not abort running kernels but freq lags — per-step switching futile.
- 30-series power sensor: board-level, ~10Hz; 30s point averages ok,
  per-kernel attribution impossible inline.
- Per-kernel visibility (if ever needed): CUPTI/torch.profiler sees
  Inductor-generated + CUDA-graph-replayed kernels; Python hooks
  (TorchDispatchMode at _model_forward, direct_register_custom_op at
  vllm/utils/torch_utils.py:901) only see eager-mode ops.
- Prior art: Zeus (github.com/ml-energy/zeus) for energy measurement +
  power-limit search; DynamoLLM for phase-aware LLM DVFS.

## State / next steps

- Branch `power-profiler` on fork Refael10ru/vllm (commit 5af1ec47e). No
  upstream PR (personal tooling; AGENTS.md fail-closed).
- [ ] On rig: `uv pip install pynvml matplotlib`, sudoers line for
      nvidia-smi (README), run `--measure-switch-latency` first (~30s).
- [ ] Full sweep (~30-60 min), eyeball heatmaps/pareto.
- [ ] Decide switching policy from switch-latency numbers.
- [ ] If decode curve flat: pick decode SM clock at pareto knee; build
      runtime applier (windowed or per-batch per measurement).
- [ ] Validation pass: mixed serving benchmark at chosen policy vs stock —
      report tok/s, TTFT/TBT, watts.
