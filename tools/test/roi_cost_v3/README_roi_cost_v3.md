# HA-ARE / ROI cost V3 on Thor

## Scope

A = V2 ROI-Batch; B = V2 Adjacency-20 (20% **per local merge**, not a global area cap). Both retain V2 padded-pixel plan ranking. The online dispatcher shares grid mapping, constructs both A and B, predicts remaining camera execution costs, and chooses B only if predicted saving exceeds the calibration margin and both shapes lie within observed training ranges. All additional planning/prediction costs are included in actual online E2E. No uncalibrated preliminary gate, no per-workload oracle B threshold. No batch splitting. One backbone and neck call per camera. FPN alignment 64.

GPU path is based on the successful V2 path, including full-image padding when required. Synthetic ROI generation is cached per case and first populated during a temporal warmup. All measurements run the real UniAD forward; no separate backbone. CUDA Events are recorded inside image feature extraction; the profiler explicitly synchronizes only after full forward (the underlying model may have implicit synchronizations). Event durations may include enqueue idle gaps. CPU timers are not added to CUDA Event durations. `roi_block_gpu_stream_ms` is a stream interval, not a standalone synchronized host wall time; E2E is the deployment metric. Raw GPU stages, host enqueue, planning, pair check counts, memory and geometry are in camera/frame CSV. Source padding, crop batch padding and merge expansion are distinct. V3 corrects V2's merge expansion field overwritten by batch_geometry summary.

This is a synthetic ROI timing experiment, not a model accuracy experiment. Crop boundaries and zero canvas change features, and downstream computation may also vary by arm. No real QC-BEV ROI generator is integrated. Shape-range fallback does not ensure memory safety; actual peaks measured. No AMP/FP16; default TF32. Keep settings fixed between collect and online.

## Files

- `profile_roi_cost_v3_thor.py`: collection and actual online execution.
- `adaptive_roi_dispatcher.py`: CPU-only cost formula and shared plan search.
- `roi_dispatch_v2_runtime.py`: local V2 geometry dependency with merge-pair counter and expansion fix. Keep in this directory, do not overwrite V2 installation.
- `fit_roi_cost_v3.py`: NumPy ridge fit, calibration, validation, online paired report (no sklearn needed).
- `test_roi_dispatch_v2_runtime.py`, `test_roi_cost_v3.py`: 13 CPU tests; pipeline fixture is synthetic only.

## Thor usage

Run from /home/uestc/uniad; unzip into tools/test/roi_cost_v3. Use fresh output directories. Existing path root is discovered using projects/configs. Run in the UniAD venv. The dataset must expose `data_infos` with `scene_token`; direct mmcv collate of a fresh dataset item avoids worker processes. Samples use preceding indices within the same scene for warmup, with temporal state reset before every arm/repeat. Dataset order assumed chronological as in UniAD val; manifest records all indices. If GPU or metadata validation fails, preserve logs and partial CSV. `COMPLETE` indicates a fully finished run; fitter rejects incomplete or duplicate data.

```bash
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH=/home/uestc/uniad:${PYTHONPATH:-}
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=11.0
export OPENBLAS_NUM_THREADS=1
set -o pipefail
(cd tools/test/roi_cost_v3 && python -m py_compile *.py && python -m unittest -v test_roi_dispatch_v2_runtime test_roi_cost_v3)
P=tools/test/roi_cost_v3
C=projects/configs/stage2_e2e/base_e2e_tf32.py
W=ckpts/uniad_base_e2e.pth
R=work_dirs/roi_cost_v3_smoke
mkdir -p "$R"
CUDA_VISIBLE_DEVICES=0 python -u "$P/profile_roi_cost_v3_thor.py" "$C" "$W" --out-dir "$R/collect" --scenes 3 --cases-per-scene 2 --shape-cases-per-train-scene 1 --repeats 1 2>&1 | tee "$R/collect.log"
python "$P/fit_roi_cost_v3.py" --input-dir "$R/collect" --out-dir "$R/fit" 2>&1 | tee "$R/fit.log"
CUDA_VISIBLE_DEVICES=0 python -u "$P/profile_roi_cost_v3_thor.py" "$C" "$W" --stage online --manifest "$R/collect/manifest.json" --model-json "$R/fit/roi_cost_model.json" --out-dir "$R/online" --repeats 1 2>&1 | tee "$R/online.log"
python "$P/fit_roi_cost_v3.py" --stage online-report --input-dir "$R/online" --out-dir "$R/report" 2>&1 | tee "$R/report.log"
```

Smoke only validates plumbing. On success repeat above with `R=work_dirs/roi_cost_v3_full`, collection flags `--scenes 30 --cases-per-scene 12 --shape-cases-per-train-scene 4 --repeats 2`, online `--repeats 2`. 18 training, 6 calibration, 6 holdout scenes. 360 ROI cases, 72 additional feasible shape-scan cases (training only), 1,584 measured collection forwards and 432 measured online forwards; one preceding warmup per measured forward by default. Shape scans include six equal-area different B/H/W configurations. ROI combinations shuffled from 6 counts × 4 area fractions × 4 layouts × 5 aspect ratios; 360 are sampled without replacement at default settings. No claim of exhaustive factorial coverage. Distinct ROI seeds per case, paired within case across arms, randomized arm order per repeat. For serious inference prefer more held-out scenes in a later run. `--no-tf32` requires an entirely new collection and fit, not mixing precisions.

## Formula

For normalized h=H/64, w=W/64, b=B, p=bhw: CNN features = p,b,h,w,bh,bw,hw,sqrt(p),p log2(b),p log2(h),p log2(w),b max(h,w),b min(h,w). Standardized ridge intercept and coefficients saved in JSON. Nonnegative predicted CNN + packing (pad and crop) + scatter form execution proxy. Packing uses crop count, crop and padded area, source pad, dimension interactions. Scatter uses crop count, crop area, dimensions. Planner model includes geometric count, candidate count and merge pair checks; these are available after plan construction. Frame models predict ROI stream interval and E2E from sums of predicted execution, estimated planner, max execution and total crops; offline diagnostics, not per-camera dispatch. Margin is one-sided P90 of overestimated A-minus-B savings on calibration scenes; heuristic, not an independent 90% statistical guarantee. Outside training range fallback A. No GPU execution duplicated for decisions. In phase 2, actual adaptive path pays both searches; compare actual E2E to both fixed arms. No second look at the holdout for tuning; if modifying formula/margin or gate, choose a new final test set.

## Results

`collect/manifest.json`, environment, CSV, COMPLETE; `fit/roi_cost_model.json`, `fit_metrics.json`, camera predictions; `online/` same CSV and environment; `report/online_metrics.json`, `online_paired.csv`. E2E mean/P90, scene-cluster bootstrap, planner cost, prediction cost, memory peaks. Only 6 test scenes by default, CI indicative; repeats/cameras not independent. Camera offline regret is against counterfactual stage-sum oracle, not actual mixed-path E2E. Use measured online results for final recommendation. If adaptive does not beat fixed A in mean and tail with meaningful scene-level evidence, retain A. For actual driving integration validate perception/trajectory accuracy and real projected ROI distributions separately.
