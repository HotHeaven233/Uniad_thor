# Thor Backbone Budget Sweep (0–100%, 5% step)

This experiment measures the complete 21-point Backbone budget curve for both
`roi_batch` and `adjacency_merge`, then generates two latency plots from the
measured CSV data. It reuses the previous `launch_bound_census_v2` packed-budget
planner and the current repository's fixed-tile / dense-adjacency implementation.

## Budget definition — unchanged from the previous pilot

The x-axis is the per-camera hard cap on the **padded CNN input area**:

\[
B\,H_{crop}\,W_{crop} \leq \frac{b}{100} H_{img}W_{img}.
\]

Padding is charged to the budget. The planner searches candidate ROI geometries
and rejects any candidate whose actual packed input exceeds the cap. This is the
same `budget_axis=packed` definition used by the previous low-budget pilot; do
not change it to `raw` when comparing curves with the earlier results.

The sweep points are:

`0, 5, 10, 15, ..., 95, 100` (21 points).

Both modes use the same scene/frame/repeat samples, the same ROI seed and layout
for a paired case, and randomized budget/mode order to reduce thermal/order bias.
Each case resets UniAD temporal state and replays the same warmup history before
its measured frame.

## 0% semantics

At 0%, no ROI is selected and ResNet/FPN are bypassed. The script returns
correctly shaped zero FPN feature tensors so the remainder of UniAD E2E can run.
The 0% point is therefore a **no-Backbone reference**, not a positive CNN workload.
It is shown separately on the plot and must not be interpreted as a launch-bound
measurement.

## What is timed

Primary plot metric: `cnn_stream_ms`, the sum of six per-camera CUDA Event
intervals around ResNet + FPN. This preserves the metric used by the earlier
pilot. CUDA Event intervals can include gaps while the CPU submits kernels; they
are not a pure sum of GPU-busy kernel durations.

The raw output also records:

- `cnn_host_ms`: host enqueue duration for ResNet + FPN; it overlaps GPU work.
- `e2e_wall_ms`: complete UniAD E2E forward wall time, synchronized once after
  the full forward.
- `raw_area_pct`, `padded_area_pct`, `budget_fill_ratio`: actual ROI and packed
  input utilization, allowing verification that the hard cap was respected.
- Per-camera crop shapes and execution signatures in `camera_shape_trace.csv`.

## Scope limitation

The experiment uses deterministic synthetic image-space ROI geometry from the
previous v2 planner, but it executes the real Thor UniAD E2E forward and the real
ResNet/FPN modules from the current `thor-deploy` code. The current repository
branch does not yet integrate the production ROI dispatcher into
`UniADTrack.extract_img_feat()`. Thus this is a controlled, paired budget sweep,
not a claim about the final production ROI generator's empirical distribution.

## Package files

```text
backbone_budget_sweep/
├── collect_backbone_budget_sweep.py
├── launch_census_utils.py
├── plot_budget_sweep.py
├── run_budget_sweep.sh
├── test_budget_sweep.py
└── README.md
```

Dependencies reused from the repository:

- `tools/test/roi_tile_dispatch_runtime.py`
- `tools/test/profile_roi_tile_dispatch_thor.py`
- `tools/test/cudagraph/shape_census_utils.py`

No model source file is modified.

## Upload to Thor

Windows PowerShell:

```powershell
scp "C:\Users\12148\Downloads\backbone_budget_sweep_thor.zip" `
  uestc@192.168.112.133:/home/uestc/uniad/tools/test/
```

Thor:

```bash
cd /home/uestc/uniad/tools/test
unzip -o backbone_budget_sweep_thor.zip
chmod +x backbone_budget_sweep/run_budget_sweep.sh

cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH="$PWD:$PWD/tools/test:$PWD/tools/test/cudagraph:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1
P=tools/test/backbone_budget_sweep
```

## Static checks

```bash
python -m py_compile "$P"/*.py
python -m unittest discover -s "$P" -p 'test_*.py' -v
bash -n "$P/run_budget_sweep.sh"
```

## Smoke test first

The smoke test checks 0%, 5%, and 100% on two scenes, one measured frame per
scene, and one repeat. It exercises the 0% bypass and the largest ROI-Batch
batch before starting the full sweep.

```bash
bash "$P/run_budget_sweep.sh" \
  projects/configs/stage2_e2e/base_e2e_tf32.py \
  ckpts/uniad_base_e2e.pth \
  work_dirs/backbone_budget_sweep_smoke \
  --sweep-budgets 0,5,100 \
  --scenes 2 --frames-per-scene 1 --repeats 1 --warmup 1 \
  2>&1 | tee work_dirs/backbone_budget_sweep_smoke.log
```

## Full 0–100% sweep

After the smoke test completes, use a fresh output directory:

```bash
bash "$P/run_budget_sweep.sh" \
  projects/configs/stage2_e2e/base_e2e_tf32.py \
  ckpts/uniad_base_e2e.pth \
  work_dirs/backbone_budget_sweep_0_100 \
  --scenes 6 --frames-per-scene 1 --repeats 2 --warmup 1 \
  2>&1 | tee work_dirs/backbone_budget_sweep_0_100.log
```

This is 21 budgets × 2 modes × 6 scenes × 1 measured frame × 2 repeats = 504
measured E2E forwards, plus 504 warmup forwards. Each point has 12 paired
measurements (6 scenes × 2 repeats).

## Outputs

- `frame_summary.csv`: per-frame CNN Stream, host enqueue, E2E wall time and
  budget utilization.
- `camera_shape_trace.csv`: per-camera actual input crop shapes and signatures.
- `budget_sweep_summary.csv`: arithmetic mean, median, P95 and standard
  deviation for each mode/budget.
- `roi_batch_budget_latency.png`: measured ROI-Batch curve.
- `adjacency_merge_budget_latency.png`: measured Adjacency-Merge curve.
- `budget_sweep_report.json`, `environment.json`, `scene_manifest.json`,
  `args.json`, `resolved_config.py`, and `COMPLETE`.

The two plots are generated automatically only after all expected samples have
been recorded. Missing budget/mode groups or an incomplete run cause an error;
there is no interpolation or fabricated point.

## Replot from measured CSV

```bash
python "$P/plot_budget_sweep.py" \
  --input-dir work_dirs/backbone_budget_sweep_0_100
```

## Transfer results back to Windows

Thor:

```bash
tar -czf /tmp/backbone_budget_sweep_results.tar.gz \
  work_dirs/backbone_budget_sweep_0_100
```

Windows PowerShell:

```powershell
scp uestc@192.168.112.133:/tmp/backbone_budget_sweep_results.tar.gz `
  "C:\Users\12148\Downloads\"
```
