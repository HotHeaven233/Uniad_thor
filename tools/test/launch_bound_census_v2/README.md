# Launch-bound-aware Backbone/FPN Shape Census v2 — Thor

## Purpose

This version narrows the next experiment to the low-budget region where the Backbone/FPN latency may become launch-bound. The budget is **Backbone input compute budget**, not BEV Query budget.

The previous pilot result supplied by the user showed:

- `adjacency_merge`: 10% -> 20% was flat (+5.99% latency); 20% -> 40% rose (+68.83%). Candidate plateau upper edge: 20%.
- `roi_batch`: 10% -> 20% rose (+35.69%); 20% -> 40% rose (+66.48%). No plateau was established at the previous low tiers.

Therefore, this v2 package supports sub-10% budgets and mode-specific pilot/census selection. It does not force the two execution modes to share a knee.

## Important scope limitation

The inspected `thor-deploy` branch does not yet contain a production integration of the real ROI-Batch / Adjacency-Merge dispatcher in `UniADTrack.extract_img_feat()`. This experiment uses deterministic synthetic image-space ROI geometry with the repository's existing fixed-tile and dense-adjacency planner, while executing the real E2E model forward and actual ResNet/FPN calls. It is a controlled workload experiment, **not** a census of the final production ROI generator. Do not describe it as production ROI distribution.

The default budget axis is `packed`: for each camera, the actual padded CNN input pixel count must satisfy `B * Hcrop * Wcrop <= budget_pct/100 * Himg * Wimg`. Padding is charged to the budget. Inputs are FP32 with TF32 enabled; AMP is disabled. No CUDA Graph is captured.

## Files

```text
launch_bound_census_v2/
├── collect_launch_shapes.py
├── launch_census_utils.py
├── run_launch_census.sh
├── run_roi_batch_low_budget.sh
├── test_launch_census.py
└── README.md
```

Dependencies already in `thor-deploy`:

- `tools/test/roi_tile_dispatch_runtime.py`
- `tools/test/profile_roi_tile_dispatch_thor.py`
- `tools/test/cudagraph/shape_census_utils.py`

No new Python dependencies are introduced.

## Install on Thor

Windows PowerShell:

```powershell
scp "C:\Users\12148\Downloads\launch_bound_census_v2.zip" uestc@192.168.112.133:/home/uestc/uniad/tools/test/
```

Thor:

```bash
cd /home/uestc/uniad/tools/test
unzip -o launch_bound_census_v2.zip
chmod +x launch_bound_census_v2/*.sh

cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH="$PWD:$PWD/tools/test:$PWD/tools/test/cudagraph:${PYTHONPATH:-}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-11.0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OPENBLAS_NUM_THREADS=1
P=tools/test/launch_bound_census_v2
R=work_dirs/launch_bound_census_v2
mkdir -p "$R"
```

## Step 1 — static checks and smoke

```bash
python -m py_compile "$P"/*.py
python -m unittest discover -s "$P" -p 'test_*.py' -v
bash -n "$P"/*.sh
bash "$P/run_launch_census.sh" smoke "$R/smoke" 2>&1 | tee "$R/smoke.log"
```

The smoke run checks one scene, one measured frame, and the six camera calls. It does not identify the knee.

## Step 2 — ROI-Batch low-budget pilot

Run only ROI-Batch at 2.5%, 5%, 10%, and 20%, on the same six-scene paired sample with two repeats:

```bash
bash "$P/run_launch_census.sh" pilot "$R/roi_batch_low_pilot" \
  --pilot-modes roi_batch \
  --pilot-budgets 2.5,5,10,20 \
  --scenes 6 --repeats 2 --warmup 1 \
  2>&1 | tee "$R/roi_batch_low_pilot.log"
```

Inspect:

```bash
cat "$R/roi_batch_low_pilot/knee_report.json"
column -s, -t "$R/roi_batch_low_pilot/pilot_summary.csv"
```

Expected interpretation:

- If 2.5 -> 5 is flat and 5 -> 10 is rising, the candidate plateau tiers are 2.5%, 5%.
- If 2.5 -> 5 and 5 -> 10 are flat but 10 -> 20 rises, the candidate plateau tiers are 2.5%, 5%, 10%.
- If the first comparison (2.5 -> 5) is already rising, do not declare a knee. Run the lower fallback below.
- If the plateau continues through 20%, the upper boundary is not bracketed; extend only if needed.

The detector requires actual packed area growth and paired per-case timing behavior; a flat time curve without meaningful input-area growth is classified as geometry-limited, not launch-bound evidence.

## Step 2b — only if no ROI-Batch plateau appears

A 64x64 crop occupies about 0.276% of a 928x1600 camera image; 0.3125% is the lowest tested budget tier with a small safety margin for that minimum crop. Probe the lower range with a fresh output directory:

```bash
bash "$P/run_launch_census.sh" pilot "$R/roi_batch_ultralow_pilot" \
  --pilot-modes roi_batch \
  --pilot-budgets 0.3125,0.625,1.25,2.5,5 \
  --scenes 6 --repeats 2 --warmup 1 \
  2>&1 | tee "$R/roi_batch_ultralow_pilot.log"
```

Use this only if the 2.5% -> 5% comparison is rising or mixed. If 0.3125% -> 0.625% is still rising, the tested geometry has not demonstrated a latency plateau above its minimum tile size; do not manufacture one.

## Step 3 — targeted Shape Census after the pilot

Adjacency-Merge already has a candidate plateau at 10%, 20%. For ROI-Batch, use exactly the `selected_budgets` reported by the new low-budget pilot. Example **only if** the ROI-Batch report selects `[2.5, 5]`:

```bash
bash "$P/run_launch_census.sh" census "$R/shape_census" \
  --mode-budgets 'roi_batch=2.5,5;adjacency_merge=10,20' \
  --scenes 12 --frames-per-scene 4 --warmup 1 \
  2>&1 | tee "$R/shape_census.log"
```

If ROI-Batch does not establish a plateau, census Adjacency-Merge only for now:

```bash
bash "$P/run_launch_census.sh" census "$R/adjacency_only_shape_census" \
  --mode-budgets 'adjacency_merge=10,20' \
  --scenes 12 --frames-per-scene 4 --warmup 1 \
  2>&1 | tee "$R/adjacency_only_shape_census.log"
```

The census traverses a real sequential E2E history for each selected mode/budget and records the actual ResNet input and ResNet/FPN output tensor signatures. It is shape collection, not a latency benchmark. It does not capture CUDA Graph.

## Knee rule

For each mode independently, compare adjacent budget tiers using paired cases. At least two thirds of cases must have >=25% actual padded-pixel growth, and at least two thirds must show either <=15% latency change (flat) or >=25% latency growth (rising). Median thresholds must also pass. A plateau must start at the minimum tested budget and include at least two tiers; a rising next tier brackets its upper edge. These are empirical heuristics, not statistical confidence intervals and not proof of CPU launch-bound behavior.

The `cnn_stream_ms` field is CUDA Event stream interval time, not the sum of GPU busy kernel time. `cnn_host_ms` is host enqueue duration and overlaps GPU work. Confirm launch-bound behavior separately with Nsight Systems by examining CUDA API tracks and GPU idle gaps inside the annotated CNN region.

## Outputs

Pilot:

- `knee_report.json`
- `pilot_summary.csv`
- `frame_summary.csv`
- `camera_shape_trace.csv`
- `environment.json`, `args.json`, `scene_manifest.json`

Census:

- `census_selection.json`
- `shape_histogram_by_budget_mode.csv`
- `shape_histogram_global.csv`
- `full_frame_topk.csv`
- `graph_signature_manifest.json`
- `census_report.json`

Always use a fresh output directory. `COMPLETE` is written only after successful completion. Graph signatures are candidates only; Phase 2 must still verify DCNv2/custom-op capture compatibility, static-buffer lifetimes, memory consumption, replay correctness, and end-to-end performance.
