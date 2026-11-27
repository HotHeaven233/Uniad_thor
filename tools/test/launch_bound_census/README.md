# Launch-bound-aware Backbone/FPN Shape Census — Thor

## Scope and budget

This replaces the earlier BEV-query-budget census. No BEV query selection or selective encoder is installed. The default fixed backbone budget is a **per-camera cap on actual packed CNN input pixels**: `B * Hcrop * Wcrop <= budget_pct / 100 * Himg * Wimg`. Padding counts toward the cap. The six available deployment tiers are 10,20,40,60,80,100. The six-camera frame cap follows by summation. The input is FP32 with TF32 enabled; no AMP or FP16.

Actual ROI selection is not integrated in the inspected Thor branch. This experiment generates deterministic synthetic image-space ROI masks (compact, spread, mixed), using the real nuScenes image tensors and existing fixed-tile/dense adjacency geometry. For the packed cap, a small CPU-only search picks a feasible candidate close to the cap, not necessarily the global optimum. Quantization and adjacency expansion can leave a budget underfilled. Both paths are tested at the same cap but may select different raw ROI masks. This is **not** a paired same-ROI accuracy or performance comparison, or a production workload census. Use `--budget-axis raw` in both stages only if the intended experiment is a nominal raw ROI union budget; this preserves the same raw ROI per case across modes, but padded CNN input can exceed the nominal budget.

The code invokes actual E2E `net(return_loss=False, rescale=True, **batch)` for every recorded frame; crop, pack, ResNet/FPN, scatter, BEVFormer and downstream heads all execute. No graph capture. The feature canvas is zero-filled outside executed crops, as in the existing ROI profiler; predictions are not valid accuracy measurements. Planner search is deliberately outside CNN timings. No claim about E2E speedup.

## Run

Unzip at `tools/test/` (archive has `launch_bound_census/` root). Dependencies from thor-deploy: `tools/test/profile_roi_tile_dispatch_thor.py`, `tools/test/roi_tile_dispatch_runtime.py`, `tools/test/cudagraph/shape_census_utils.py`. No new Python dependencies. Run from `/home/uestc/uniad` in the existing venv.

```bash
P=tools/test/launch_bound_census
export PYTHONPATH="$PWD:$PWD/tools/test:$PWD/tools/test/cudagraph:${PYTHONPATH:-}"
python -m py_compile "$P"/*.py
python -m unittest discover -s "$P" -p 'test_*.py' -v
bash -n "$P/run_launch_census.sh"
R=work_dirs/launch_bound_census
mkdir -p "$R"
set -o pipefail
bash "$P/run_launch_census.sh" smoke "$R/smoke" 2>&1 | tee "$R/smoke.log"
bash "$P/run_launch_census.sh" pilot "$R/pilot" --scenes 6 --repeats 2 2>&1 | tee "$R/pilot.log"
cat "$R/pilot/knee_report.json"
bash "$P/run_launch_census.sh" census "$R/census" --pilot-report "$R/pilot/knee_report.json" --scenes 12 --frames-per-scene 4 2>&1 | tee "$R/census.log"
```

Smoke: one scene, one measured frame, six conditions, one preceding warmup each. Full pilot: six scenes, one measured frame each, two repeats; initial 10/20/40 x two paths = 72 measured + 72 warmup forwards. More budgets are only tested for a path whose plateau continues through the highest tested tier. Set `--max-pilot-budget 40` to prohibit extensions. The pilot runs real paired cases, randomizes order and resets temporal state before each arm. Warmup uses the same ROI geometry as the corresponding measured frame. Census uses 12 scenes and four sequential measured frames per scene by default, one preceding warmup, and only the selected tiers for each path. `--scenes 0 --frames-per-scene 0` covers all eligible scene frames after the warmup prefix. The census is shape-only, not a latency benchmark; E2E wall time is a diagnostic.

## Knee detection

For each path, compute the median repeated six-camera sum of Backbone+FPN CUDA Event stream intervals per case. Compare adjacent budget tiers using paired cases. At least 2/3 of cases must exhibit at least 25% growth in actual packed pixels, and at least 2/3 must also show either absolute latency growth <=15% (flat) or latency growth >=25% (rising). The median must also satisfy the timing threshold. The plateau must start at the minimum tested budget and contain at least two tiers. A rising next tier brackets it. Only plateau tiers are selected automatically; the first rising upper tier is pilot-only. Mixed/geometry-limited upper comparisons yield a conservative plateau prefix if any. If no low-tier plateau is established, the path is omitted; if both paths fail, census stops rather than manufacturing a knee. If the plateau never ends, the report states that no upper boundary was established. Thresholds are heuristics, not confidence intervals, and can be changed in pilot CLI. `--force-budgets 10,20` on census is an explicit exploratory override, applied to both modes; not a confirmed knee. A smoke report cannot automatically select final census tiers. If 10% is already above the knee, the original fixed budget grid is insufficient to measure the plateau; lower tiers require changing the workload specification.

CUDA Events measure stream intervals, not summed kernel busy time. Host enqueue `perf_counter` is diagnostic only; it overlaps GPU work. An apparent plateau does not prove CPU launch bound. Use optional `--nvtx` to mark CNN regions for a small separate Nsight Systems run if available. For example:

```bash
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
  -o "$R/nsys_knee" bash "$P/run_launch_census.sh" pilot "$R/trace" \
  --scenes 1 --repeats 1 --max-pilot-budget 40 --nvtx
```

Compare GPU gaps and CUDA API/kernel tracks within annotated CNN regions; the trace run is not a clean pilot and cannot auto-select final tiers.

## Outputs

Pilot: `pilot_summary.csv`, `knee_report.json`, `frame_summary.csv`, `camera_shape_trace.csv`. Both stages record raw ROI pixels, crop pixels, actual packed pixels, budget fill, shapes and path labels; camera rows include actual execution rectangles. Census: `census_selection.json`, `shape_histogram_by_budget_mode.csv`, `shape_histogram_global.csv`, `full_frame_topk.csv`, `graph_signature_manifest.json`, `census_report.json`, raw traces. Signatures contain the actual input/output structures, tensor metadata and execution flags; budget and path are not part of the graph key. Top-K is invocation coverage, with a separate per-scope all-six-call frame coverage. Global frequencies are unweighted enumerated occurrence frequencies, not expected deployment probabilities. Graph-candidate signatures are not evidence of capture compatibility. Verify DCNv2, buffer lifetimes, replay correctness, memory, and performance in Phase 2.

`COMPLETE` is written only after successful completion. Use fresh directories; CSV streams are flushed per measured forward. The census checks pilot environment/source/config/weight metadata for consistency. No GPU execution was performed by the author of this package; CPU tests and syntax checks passed with the existing geometry source. Run smoke on Thor before full pilot.
