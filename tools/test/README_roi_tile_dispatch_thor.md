# UniAD Thor：Fixed-Tile ROI-Batch vs Adjacency-Merge Profiling

本实验按已确认的固定裁剪单元方案实现，放在：

```text
/home/uestc/uniad/tools/test/
```

文件：

- `profile_roi_tile_dispatch_thor.py`：真实 UniAD E2E Forward 内的实验脚本。
- `roi_tile_dispatch_runtime.py`：固定单元映射、邻接合并和运行时成本模型选择器。
- `test_roi_tile_dispatch.py`：不依赖 GPU 的几何逻辑单元测试。

## 1. 实验目标和数据来源

本实验**不使用 BEV Query 数量生成 ROI**。脚本在 nuScenes 的真实六路相机图像上生成可控的 image-space ROI 矩形，目的是独立研究 ROI 数量、覆盖面积、重叠和空间分布对两种计算路径的影响。

模型仍然走实际执行链：

```text
UniAD.forward(return_loss=False)
  -> UniAD.forward_test()
  -> UniADTrack.simple_test_track()
  -> UniADTrack.get_bevs()
  -> patched UniADTrack.extract_img_feat()
  -> 实际 img_backbone (ResNet-101)
  -> 实际 img_neck (FPN)
  -> 恢复完整尺寸多尺度特征图
  -> BEVFormer Encoder
  -> Tracking / Mapping / Motion / Occupancy / Planning
```

脚本只在运行时替换 `extract_img_feat` 的图像特征提取逻辑；不会另起一个脱离 UniAD 的 ResNet benchmark。完整 E2E Forward 仍然执行。两种策略使用相同的图像帧、ROI 几何、模型权重和 TF32 设置，分别运行后配对比较。

**注意：这是延迟实验，不是精度实验。** 未计算的图像区域对应的特征图以零填充；裁剪边界也缺少完整图像的感受野上下文。不能用该脚本的输出声称检测或规划精度保持不变。

## 2. 固定裁剪单元定义

对每个摄像头独立计算：

```text
max_roi_w = 所有原始 ROI 的最大宽度
max_roi_h = 所有原始 ROI 的最大高度
unit_w_32 = ceil(max_roi_w / 32) * 32
unit_h_32 = ceil(max_roi_h / 32) * 32
```

然后将实际输入单元尺寸进一步向上对齐到 `lcm(32, 64)=64` 的倍数：

```text
unit_w = ceil(unit_w_32 / 64) * 64
unit_h = ceil(unit_h_32 / 64) * 64
```

原因：UniAD 当前 FPN 的四个输出步长为 8、16、32、64。裁剪单元的宽高和网格原点必须能被最粗的 64 像素步长整除，否则 stride-64 特征无法在原始特征图上整数索引对齐。64 是 32 的倍数，因此仍满足 32 对齐，只是可能比最小 32 对齐尺寸稍大。CSV 同时保存 `unit_w_32/unit_h_32` 和最终 `unit_w/unit_h`，便于检查扩张量。

以图像左上角为网格原点，构造不重叠固定单元网格。图像右侧/下侧不足一个单元的部分在输入端补零，保证所有裁剪单元形状一致。

每个 ROI 激活**所有与其有正面积相交**的固定单元。坐标采用半开区间 `[x0,x1) × [y0,y1)`，刚好贴边但没有像素交集不算相交。多个 ROI 激活同一个单元时，通过 tile ID 集合去重；该单元只进入 ResNet 一次。

## 3. 两种计算路径

### A. Fixed-Tile ROI-Batch

1. ROI 映射到所有相交固定单元。
2. 对重复 tile ID 去重。
3. 所有唯一 tile 裁剪为相同的 `[3, unit_h, unit_w]`。
4. 每个摄像头把所有 tile 一次性拼成一个 Tensor：`[K, 3, unit_h, unit_w]`。
5. 每个摄像头只调用一次 ResNet 和一次 FPN；不做 batch 分块。
6. 将多尺度特征放回完整图像特征画布。

这里 ResNet 的 batch size 是唯一 tile 数量 K；K 越大，计算量和显存需求可能越大，但 ResNet 调用次数不随 K 增加。六路相机分别执行，每个 E2E 帧固定最多 6 次 ResNet/FPN 调用。

### B. Adjacency-Merge

1. 使用完全相同的 ROI -> tile 映射和去重结果。
2. 邻接矩阵的节点是**去重后的固定 tile**，不是原始 ROI。
3. 只有两个 tile 共享完整水平或垂直边界时才连边；对角接触不算邻接。
4. 用连通分量得到 tile 组，每组取最小外接矩形。
5. 如果不同连通分量的外接矩形仍有正面积重叠，则继续合并这些外接矩形，避免同一像素被送入 ResNet 重复计算。此步骤单独记录为 `bbox_overlap_merge_cpu_ms`。
6. 将最终矩形裁剪成一个 Batch。矩形大小不同时，统一 Padding 到该摄像头当前 Batch 的最大宽高。
7. 每个摄像头仍然只调用一次 ResNet 和一次 FPN；不做 batch 分块。
8. 将多尺度特征放回完整图像特征画布。

Adjacency-Merge 的潜在收益是减少 Batch 样本数；潜在代价是连通分量的外接矩形可能覆盖未激活 tile，且不同矩形 Padding 到最大尺寸时会产生额外计算。两者都被统计。

## 4. 横坐标与面积定义

命令行 `--area-fractions` 只是生成原始 ROI 时的目标参数，表示**原始 ROI 矩形面积之和**相对于单张图像面积的目标值。由于 ROI 可能重叠、边界裁剪和尺寸取整，实际值可能不同。

最终分析不使用目标值，而使用 CSV 中实测的三个面积：

1. `raw_roi_union_area_ratio_pct_all_cameras`：原始 ROI 的精确并集面积 / 六路图像总面积。它不重复计算重叠像素，是独立于 Query 的主要横坐标。
2. `unique_tile_input_area_ratio_pct_all_cameras`：ROI-Batch 实际送入 ResNet 的唯一固定单元总面积 / 六路图像总面积。
3. `executed_crop_area_ratio_pct_all_cameras`：当前策略实际裁剪矩形面积总和 / 六路图像总面积。对 Adjacency-Merge，它可能因外接矩形而大于唯一 tile 面积。

因此可以分别看到“原始 ROI 覆盖了多少图像”“固定单元量化后需要计算多少像素”“邻接合并额外扩张了多少面积”。

## 5. 时延组成与决策目标

每帧记录以下组件（CPU host time 和 CUDA event time 分开）：

- `tile_mapping_cpu_ms`：ROI 到固定单元映射和去重。
- `adjacency_matrix_cpu_ms`：显式稠密邻接矩阵分配与构建；ROI-Batch 为 0。
- `connected_components_cpu_ms`：连通分量与初始外接矩形生成；ROI-Batch 为 0。
- `bbox_overlap_merge_cpu_ms`：重叠外接矩形去重；ROI-Batch 为 0。
- `tile_pad_host_ms/tile_pad_gpu_ms`：构造图像边界补零画布。
- `feature_canvas_init_gpu_ms`：初始化完整多尺度特征画布。
- `crop_pack_host_ms/crop_pack_gpu_ms`：裁剪、分配和 Batch 拼接/Padding。
- `resnet_gpu_ms`：实际 ResNet-101 CUDA event 时间。
- `fpn_gpu_ms`：实际 FPN CUDA event 时间。
- `scatter_gpu_ms`：将裁剪特征写回完整特征画布。
- `roi_block_wall_ms`：覆盖六路相机、从 ROI 映射开始到最后一次 CUDA 同步完成的实测 wall time。
- `e2e_forward_ms`：完整 UniAD E2E Forward wall time。

**运行时决策的监督目标是 `roi_block_wall_ms`，而不是把各个组件时间简单相加。** CPU 提交和 GPU 执行可能存在异步/重叠，组件之和只用于解释开销构成；脚本额外输出 `component_sum_to_wall_ratio` 供诊断，不把它当成严格的 wall-time 分解。

脚本在 ROI 特征提取块的末尾做一次 CUDA 同步，以便得到从 tile 映射开始到全部 GPU 计算完成的真实 ROI-block wall time；因此 `e2e_forward_ms` 是带该 profiling 同步点的 instrumented E2E 时间，主要用于确认完整下游链路正常执行，不应直接当作未插桩生产推理的绝对延迟。两种模式采用完全相同的同步测量边界。

脚本分别拟合 ROI-Batch 和 Adjacency-Merge 的 Ridge 成本模型。训练/留出集按 `workload_id` 切分，避免同一 ROI 几何的重复帧同时进入训练和测试。留出集报告决策准确率、平均选择 regret 和 P90 regret。最终模型用全部样本重新拟合，写入 JSON。

运行时 `choose_roi_execution_mode()` 只使用原始 ROI 几何、tile 映射和 O(K) 邻接边数量估计，不提前分配稠密邻接矩阵；随后比较两个预测 wall time，并可使用 3% hysteresis 避免频繁切换。

## 6. Windows PowerShell：上传到 Thor

将下载的文件放在：

```text
C:\Users\12148\Downloads\
```

PowerShell 中执行：

```powershell
ssh uestc@192.168.112.133 "mkdir -p /home/uestc/uniad/tools/test"
```

上传四个文件：

```powershell
scp `
  "C:\Users\12148\Downloads\profile_roi_tile_dispatch_thor.py" `
  "C:\Users\12148\Downloads\roi_tile_dispatch_runtime.py" `
  "C:\Users\12148\Downloads\test_roi_tile_dispatch.py" `
  "C:\Users\12148\Downloads\README_roi_tile_dispatch_thor.md" `
  uestc@192.168.112.133:/home/uestc/uniad/tools/test/
```

也可以只上传 ZIP：

```powershell
scp "C:\Users\12148\Downloads\roi_tile_dispatch_thor_v1.zip" uestc@192.168.112.133:/home/uestc/uniad/tools/test/
```

Thor 上解压：

```bash
cd /home/uestc/uniad/tools/test
unzip -o roi_tile_dispatch_thor_v1.zip
```

## 7. Thor 环境检查

```bash
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH=/home/uestc/uniad:${PYTHONPATH:-}
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="11.0"
```

语法检查：

```bash
python -m py_compile \
  tools/test/profile_roi_tile_dispatch_thor.py \
  tools/test/roi_tile_dispatch_runtime.py \
  tools/test/test_roi_tile_dispatch.py
```

CPU-only 几何单元测试：

```bash
cd /home/uestc/uniad/tools/test
python -m unittest -v test_roi_tile_dispatch.py
cd /home/uestc/uniad
```

## 8. Smoke Test

先跑 8 个小规模 workload，每个 workload 每种模式测 1 帧，另有 1 帧 warmup：

```bash
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH=/home/uestc/uniad:${PYTHONPATH:-}

CUDA_VISIBLE_DEVICES=0 python tools/test/profile_roi_tile_dispatch_thor.py \
  projects/configs/stage2_e2e/base_e2e_tf32.py \
  ckpts/uniad_base_e2e.pth \
  --out-dir work_dirs/roi_tile_dispatch_smoke \
  --roi-counts 2,4 \
  --area-fractions 0.05,0.15 \
  --layouts spread,cluster \
  --aspect-ratios 1.0 \
  --frames-per-workload 1 \
  --warmup-frames 1 \
  --workers 0 \
  --max-workloads 8
```

Smoke Test 重点检查：

- `unique_tile_count` 是否小于等于 `roi_tile_hits`，且重叠 ROI 激活的 tile 已去重。
- 两种模式的 `raw_roi_union_area_ratio_pct_all_cameras` 和 `unique_tile_input_area_ratio_pct_all_cameras` 是否一致。
- ROI-Batch 每摄像头 `exec_crop_count == unique_tile_count`，`batch_padding_waste_pct == 0`。
- 两种模式每摄像头均为一次 ResNet 和一次 FPN 调用。
- Adjacency-Merge 的邻接矩阵只连接共享边界 tile；其合并扩张面积和 Batch Padding 有记录。
- E2E Forward 能继续执行到下游任务并正常返回。
- 生成 CSV、PNG 和 `roi_tile_dispatch_model.json`。

## 9. 正式实验

先跑固定长宽比的主实验，观察 ROI 数量、面积和空间布局的主效应：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test/profile_roi_tile_dispatch_thor.py \
  projects/configs/stage2_e2e/base_e2e_tf32.py \
  ckpts/uniad_base_e2e.pth \
  --out-dir work_dirs/roi_tile_dispatch_thor_tf32 \
  --roi-counts 2,4,8,16 \
  --area-fractions 0.05,0.10,0.20 \
  --layouts spread,cluster,chain \
  --aspect-ratios 1.0 \
  --frames-per-workload 4 \
  --warmup-frames 2 \
  --workers 0
```

再单独研究长宽比影响：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test/profile_roi_tile_dispatch_thor.py \
  projects/configs/stage2_e2e/base_e2e_tf32.py \
  ckpts/uniad_base_e2e.pth \
  --out-dir work_dirs/roi_tile_dispatch_aspect_tf32 \
  --roi-counts 4,8,16 \
  --area-fractions 0.10,0.20 \
  --layouts spread,cluster,random \
  --aspect-ratios 0.5,1.0,2.0 \
  --frames-per-workload 4 \
  --warmup-frames 2 \
  --workers 0
```

FP32 实验需要单独运行并保存到另一个目录，命令追加 `--no-tf32`。不要把 TF32 和 FP32 的样本混在同一个成本模型里。

## 10. 输出文件

- `roi_tile_camera_raw.csv`：逐帧、逐相机的 ROI、tile、邻接图、Batch 形状和组件时延。
- `roi_tile_frame_raw.csv`：六相机汇总时延、面积和成本模型特征。
- `roi_tile_paired.csv`：相同 workload/frame 的两种模式配对比较。
- `roi_tile_summary.csv`：按 ROI 数量、面积目标、布局、长宽比聚合的均值/中位数/P90。
- `roi_tile_dispatch_model.json`：最终两种模式的总 wall-time 模型和组件成本模型。
- `roi_tile_holdout_metrics.json`：按 workload 隔离的留出集决策指标。
- `roi_tile_holdout_paired.csv`：留出集逐样本预测与真实胜出模式。
- `roi_wall_vs_raw_union_area.png`：总 ROI wall time vs 原始 ROI 并集面积。
- `roi_wall_vs_tile_input_area.png`：总 ROI wall time vs ROI-Batch 唯一固定单元输入面积。
- `roi_latency_components.png`：各时延组成 vs 固定 tile 输入面积。
- `adjacency_merge_expansion.png`：邻接边数量与合并扩张面积。
- `roi_mode_winner_counts.png`：两种模式实际胜出次数。
- `paired_mode_latency_difference.png`：配对时延差，负数代表 Adjacency-Merge 更快。

## 11. 运行时选择器示例

把 `roi_tile_dispatch_model.json` 和 `roi_tile_dispatch_runtime.py` 放在同一 Python import 路径中。真实调度器提供每个相机的原始 ROI box 列表（图像像素坐标，半开区间）：

```python
from roi_tile_dispatch_runtime import (
    load_dispatch_profile,
    choose_roi_execution_mode,
    build_execution_plan,
)

profile = load_dispatch_profile(
    "work_dirs/roi_tile_dispatch_thor_tf32/roi_tile_dispatch_model.json"
)

# camera_roi_boxes: list of 6 cameras; each camera is a list of (x0,y0,x1,y1)
# image_h/image_w must match the benchmark image resolution.
decision = choose_roi_execution_mode(
    profile,
    camera_roi_boxes,
    image_h=928,
    image_w=1600,
    current_mode=None,
    switch_margin=0.03,
)

selected_mode = decision["selected_mode"]
for camera_geometry in decision["camera_geometries"]:
    plan = build_execution_plan(camera_geometry, selected_mode)
    camera_crop_boxes = plan["exec_boxes"]
```

示例中的 `image_h=928,image_w=1600` 仅适用于当前 config 的实际张量尺寸；应以运行日志/CSV 的 `image_h/image_w` 为准。部署时必须使用相同的模型权重、输入分辨率、FPN 配置和 TF32 精度重新标定。合成 ROI workload 上拟合的模型先用于研究和初步验证；要用于正式部署决策，还应以真实 scheduler 输出的 ROI 分布补充采样并重新标定。
