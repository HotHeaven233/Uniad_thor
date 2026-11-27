# UniAD Thor — Phase 1: Real E2E Backbone/FPN Shape Census

## Purpose

This phase runs the configured UniAD test dataset through the actual E2E inference forward and records every real call to `model.img_backbone` and `model.img_neck`. It does **not** synthesize ROIs, change image tensors, capture CUDA Graphs, or change model parameters.

The output is a workload manifest for Phase 2 Graph pre-capture planning. A manifest entry is keyed by the exact backbone input tensor structure/shape/stride/dtype/device/layout and the observed ResNet/FPN output structures. Path labels are kept as metadata and do not split an otherwise identical graph signature.

## Important repository-specific caveat

In the `thor-deploy` source inspected for this package, `UniADTrack.extract_img_feat()` reshapes the image tensor from `[B,N,C,H,W]` to `[B*N,C,H,W]` and invokes `img_backbone` once. The ROI-Batch/Adjacency implementations in the earlier profiling scripts are runtime benchmark patches and use generated/synthetic ROI workloads; they are not automatically activated by `base_e2e_tf32.py` or by the native model source.

Therefore:

- Running the default command records the shapes of the **currently configured native E2E test path**.
- If that path is still full-frame, expect one full-frame backbone call per frame (typically batch `B*Ncam`, with H/W determined by test pipeline padding).
- This default run does **not** predict future ROI-Batch/Adjacency shapes unless the corresponding real ROI execution implementation is already integrated into the model or supplied through a runtime adapter.
- Do not use the synthetic ROI workloads from `profile_roi_dispatch_v2_thor.py` as if they were real test-set ROI shapes.

If a real runtime ROI implementation is installed separately, pass an adapter file. Adapter contract:

```python
# real_roi_census_adapter.py

def install(model, cfg):
    # Install the real runtime path used in the intended test deployment.
    # Do not generate synthetic ROI boxes here.
    pass

def before_forward(model, frame_info):
    # Optional: clear per-frame adapter metadata / set the frame context.
    pass

def get_backbone_call_metadata(model, frame_info):
    # Optional. Return one dict per img_backbone invocation, in call order.
    # Example: [{'path': 'roi_batch', 'camera': 0, 'tile_count': 3}, ...]
    # Leave this function out if the runtime stores rows in a supported model attribute.
    return None
```

## Output files

- `frame_summary.csv`: one row per test frame, including number of ResNet calls.
- `frame_shape_trace.csv`: one row per actual ResNet invocation, including input, ResNet output, FPN input/output, exact signature ID, and optional path metadata.
- `shape_histogram.csv`: all unique exact execution signatures sorted by frequency.
- `path_shape_summary.csv`: signature frequencies split by path label when available.
- `shape_manifest.json`: canonical candidate graph manifest plus hardware/software/model context.
- `census_report.json`: dataset coverage, unique signature count, top-K coverage, scene transitions, and warnings.
- `environment.json`, `resolved_config.py`, `COMPLETE`: reproducibility information and successful completion marker.

## Install on Thor

Copy the package directory to:

`/home/uestc/uniad/tools/test/cudagraph/`

Then:

```bash
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH=/home/uestc/uniad:${PYTHONPATH:-}
P=tools/test/cudagraph
python -m py_compile "$P"/*.py
(cd "$P" && python -m unittest -v test_shape_census.py)
bash -n "$P/run_shape_census.sh"
```

## Run smoke test

```bash
bash tools/test/cudagraph/run_shape_census.sh smoke
```

This processes only the first 24 dataset entries. It validates loading, actual E2E invocation, hooks, and CSV output; it is **not** a complete shape census and must not be used to decide final Graph coverage.

## Run full test-set census

```bash
bash tools/test/cudagraph/run_shape_census.sh full
```

Defaults:

- Config: `projects/configs/stage2_e2e/base_e2e_tf32.py`
- Checkpoint: `ckpts/uniad_base_e2e.pth`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- FP32 tensors with TF32 enabled, no AMP/FP16
- dataset order, `samples_per_gpu=1`, real sequential E2E state
- one CUDA synchronize after each frame (census only; this is not a latency benchmark)

Override paths:

```bash
CONFIG=projects/configs/stage2_e2e/base_e2e_tf32.py \
CHECKPOINT=ckpts/uniad_base_e2e.pth \
bash tools/test/cudagraph/run_shape_census.sh full work_dirs/cudagraph_shape_census_full
```

If a real adapter is required:

```bash
bash tools/test/cudagraph/run_shape_census.sh full \
  work_dirs/cudagraph_shape_census_roi \
  --adapter tools/test/cudagraph/real_roi_census_adapter.py
```

The output directory must be fresh. Partial CSV files are flushed on failure/interrupt, but `COMPLETE` is written only after the full census succeeds. Do not reuse a partial directory; use a new output directory or inspect and resume manually after confirming model temporal-state warmup.

## Interpretation for Phase 2

`shape_manifest.json` identifies candidate exact signatures; it does not certify that each path is CUDA-Graph capturable. Before building the Graph pool, separately test each frequent signature for:

1. fixed static input/output buffer addresses and lifetimes;
2. ResNet/ FPN custom-op compatibility (especially DCNv2);
3. no CPU-side dynamic allocation or unsupported synchronization during capture;
4. stable output tensor structure and aliasing;
5. graph memory pool and static-buffer memory cost;
6. replay correctness versus Eager on the same input tensor.

The census itself deliberately does not call `torch.cuda.CUDAGraph()`.
