# UniAD Thor ROI Dispatch V2：第二轮真实 E2E 性能实验

## 1. 实验目标

V2 用于在 NVIDIA Jetson AGX Thor 上，比较 ROI-Batch 与稀疏邻接/矩形合并策略的**真实端到端推理开销**。实验只替换 UniAD 的 `extract_img_feat` 图像特征提取实现；模型仍通过原始 `model(return_loss=False, rescale=True, **data)` 执行，后续 BEVFormer、tracking、mapping、motion、planning 等网络路径继续运行。

ROI 是在真实 nuScenes 相机图像张量上生成的合成图像坐标 ROI，与 BEV Query 数量解耦。未覆盖的特征画布区域填零，因此这是一项**性能实验，不是精度实验**；不能用本实验输出评价检测/规划精度。

## 2. V2 对比的执行臂

默认执行以下六种策略：

| arm | 含义 |
|---|---|
| `v1_roi_batch` | V1 基线：最大 ROI 宽高决定统一 tile，原点固定为 (0,0) |
| `v2_roi_batch` | V2：P50/P75/P90/P100 尺寸候选 + x/y 相位搜索，选择 Batch 输入像素代理最小的候选 |
| `adjacency_0` | V2：稀疏四邻接 + Union-Find + 精确矩形分解，不允许额外面积膨胀 |
| `adjacency_05` | 在精确矩形基础上，允许单次局部合并最多 5% 面积扩张 |
| `adjacency_10` | 允许单次局部合并最多 10% 面积扩张 |
| `adjacency_20` | 允许单次局部合并最多 20% 面积扩张 |

每个 arm 每个摄像头仍然只执行一次 ResNet 和一次 FPN；不拆分 Batch。六路相机分别规划、分别提取，再恢复到完整特征画布。

V2 候选尺寸的宽度、高度分别根据 ROI 尺寸分位数生成。候选相位只尝试 0 和不超过半个 tile 的 64 像素对齐偏移。网格边缘 tile 可以比名义 tile 小，但所有 crop 起点、宽度、高度均为 64 的整数倍，满足 FPN stride 8/16/32/64 的特征坐标对齐。

## 3. V2 的算法与成本范围

### CPU 规划成本

- ROI 尺寸分位数统计与候选规格生成；
- 每个候选的 ROI→tile 映射、相交单元激活和去重；
- 候选相位/尺寸搜索与最终候选选择；
- 稀疏四邻接构建（只检查右、下邻居）；
- Union-Find 连通分量；
- Scanline 精确矩形分解；
- 成本约束的局部矩形合并；
- 合并候选的面积、重叠、Batch Padding 评估。

### GPU / 数据搬运成本

- source canvas 边界补零；
- Crop Batch 分配、复制和 Batch Padding；
- ResNet；
- FPN；
- Feature Canvas 分配/初始化；
- 特征 Scatter 恢复。

### 显存成本

记录完整 E2E forward 的峰值 allocated/reserved 显存，以及每路相机的 Batch Tensor 字节数和 Feature Canvas 字节数。`batch_padding_area_px`、`tile_input_outside_image_area_px`、`exec_crop_outside_image_area_px` 分开记录，避免把图像边界补零和 Batch Padding 混为一谈。

### 计时规则

**`extract_img_feat` 内不调用 `torch.cuda.synchronize()`。** 每个 GPU 阶段用 CUDA Event 记录；CPU 规划和 Python enqueue 用 `perf_counter` 记录。每个完整 UniAD forward 结束后只同步一次，再读取事件时间。

主要决策指标是 `e2e_forward_ms`：完整 `model(return_loss=False, ...)` 调用加一次 forward 结束同步的实际 Wall Time。`roi_block_gpu_stream_ms` 是 GPU Stream 区间诊断值，`roi_block_host_enqueue_ms` 是主机端 enqueue/规划时间；二者可能重叠，**不能相加**，组件时间也不能简单相加来代替 E2E。

合成 ROI 生成只在每个 workload 第一次进入 forward 时建立并缓存；warmup 会完成该步骤，正式测量帧不会重复计算随机 ROI 生成。它属于实验数据生成器，不是被测 ROI 路径的一部分。ROI 映射、候选搜索、邻接、分解、合并等规划成本均在每次测量 forward 内重新计算并计时。

## 4. 文件

- `profile_roi_dispatch_v2_thor.py`：Thor 真实 E2E 实验脚本。
- `roi_dispatch_v2_runtime.py`：纯 CPU 几何、稀疏邻接、矩形分解和候选搜索模块。
- `test_roi_dispatch_v2_runtime.py`：CPU 单元测试。
- `README_roi_dispatch_v2_thor.md`：本说明。

## 5. Windows PowerShell 上传

下载 ZIP 到 `C:\Users\12148\Downloads\` 后执行：

```powershell
ssh uestc@192.168.112.133 "mkdir -p /home/uestc/uniad/tools/test/roi_dispatch_v2"
scp "C:\Users\12148\Downloads\roi_dispatch_v2_thor_v2.zip" uestc@192.168.112.133:/home/uestc/uniad/tools/test/roi_dispatch_v2/
```

## 6. Thor 环境与解压

```bash
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH=/home/uestc/uniad:${PYTHONPATH:-}
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="11.0"

cd /home/uestc/uniad/tools/test/roi_dispatch_v2
unzip -o roi_dispatch_v2_thor_v2.zip
python -m py_compile profile_roi_dispatch_v2_thor.py roi_dispatch_v2_runtime.py test_roi_dispatch_v2_runtime.py
python -m unittest -v test_roi_dispatch_v2_runtime.py
```

8 项 CPU 几何/规划测试应全部通过。测试只验证几何算法，不代表 Thor GPU E2E 已验证。

## 7. Smoke Test

先运行小规模测试，确认 plugin hook、数据读取、FPN 特征恢复和 E2E forward 正常：

```bash
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH=/home/uestc/uniad:${PYTHONPATH:-}

mkdir -p work_dirs/roi_dispatch_v2_smoke
CUDA_VISIBLE_DEVICES=0 python tools/test/roi_dispatch_v2/profile_roi_dispatch_v2_thor.py \
  projects/configs/stage2_e2e/base_e2e_tf32.py \
  ckpts/uniad_base_e2e.pth \
  --out-dir work_dirs/roi_dispatch_v2_smoke \
  --roi-counts 4 \
  --area-fractions 0.05 \
  --layouts cluster,spread \
  --aspect-ratios 1.0 \
  --frames-per-workload 1 \
  --warmup-frames 1 \
  --workers 0 \
  2>&1 | tee work_dirs/roi_dispatch_v2_smoke/run.log
```

Smoke Test 完成后检查：

- `roi_v2_frame_components.csv` 是否包含所有 arm；
- `roi_v2_camera_components.csv` 是否每帧每个 arm 有 6 路相机记录；
- `roi_v2_candidates.csv` 是否包含候选搜索记录；
- `roi_v2_decision.json` 是否生成；
- E2E forward 是否正常完成，没有 CUDA/FPN 尺寸错误。

## 8. 正式 TF32 实验

建议先用当前 TF32 配置进行主实验。默认 3 个 ROI 数量 × 2 个面积比例 × 4 种布局 × 3 种长宽比，共 72 个 workload；每个 workload 测 6 个 arm，每个 arm 1 个 warmup + 2 个测量帧。若时间有限，可先把 `--aspect-ratios` 改为 `1.0` 做 24 个 workload 的主筛选，再单独做长宽比压力测试。

```bash
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH=/home/uestc/uniad:${PYTHONPATH:-}
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="11.0"

mkdir -p work_dirs/roi_dispatch_v2_thor_tf32
CUDA_VISIBLE_DEVICES=0 python tools/test/roi_dispatch_v2/profile_roi_dispatch_v2_thor.py \
  projects/configs/stage2_e2e/base_e2e_tf32.py \
  ckpts/uniad_base_e2e.pth \
  --out-dir work_dirs/roi_dispatch_v2_thor_tf32 \
  --roi-counts 4,8,16 \
  --area-fractions 0.05,0.10 \
  --layouts spread,cluster,chain,random \
  --aspect-ratios 0.5,1.0,2.0 \
  --frames-per-workload 2 \
  --warmup-frames 1 \
  --workers 0 \
  2>&1 | tee work_dirs/roi_dispatch_v2_thor_tf32/run.log
```

本实验默认 FP32 Tensor + TF32 允许（由 `base_e2e_tf32.py` 配置和 PyTorch backend flag 决定）。若要做纯 FP32 对照，使用同一 workload/seed/arm 命令并追加 `--no-tf32`，输出到另一个目录；不要把两种精度的样本混在同一统计表中。

## 9. 自动判定规则

脚本不会只按所有帧的总平均值下结论，而是先对每个 `workload_id` 的重复测量帧取中位数，以 workload 作为独立统计单位。

1. 先比较 V1 与 V2 ROI-Batch，选择全局更可信的 ROI-Batch 几何版本；V2 只有在 workload 胜率超过 50% 且 `V1 - V2` 的 95% bootstrap CI 下界大于 0 时才替代 V1。
2. 每个 workload 内，从 `adjacency_0/05/10/20` 中选实测 E2E 最快的一种，作为对 ROI-Batch 的强对手。
3. ROI-Batch 若在至少 80% workload 中获胜，且 `best adjacency - ROI-Batch` 平均差的 95% workload-bootstrap CI 下界大于 0，判定为 `ROI_BATCH_DOMINATES_USE_ROI_BATCH`。
4. 若胜率超过 50% 但不足 80%，且 CI 下界大于 0，判定为 `ROI_BATCH_MAJORITY_KEEP_ROI_BATCH_DEFAULT`。
5. 否则不部署自动 dispatcher；保留 ROI-Batch 安全基线，继续按布局/面积/长宽比分析。

注意：每个 workload 里选最快 adjacency arm 对 ROI-Batch 是偏保守的比较，因为 adjacency 获得了四种 expansion 参数中的最优结果。bootstrap 在 workload 级别重采样，避免把同一 ROI 几何下的重复帧错误地当作完全独立样本。

## 10. 输出文件

| 文件 | 内容 |
|---|---|
| `roi_v2_camera_components.csv` | 每摄像头规划、Batch、ResNet/FPN、Scatter、内存相关数据 |
| `roi_v2_frame_components.csv` | 每帧完整 E2E、GPU Stream、CPU 规划汇总和显存峰值 |
| `roi_v2_candidates.csv` | 每摄像头每个候选尺寸/相位的几何、Padding 和搜索时间 |
| `roi_v2_paired_e2e.csv` | 同一 workload/frame 下 ROI-Batch 与各 adjacency arm 配对差值 |
| `roi_v2_summary.csv` | 按 arm、布局、ROI 数量、面积和长宽比汇总 |
| `roi_v2_decision.json` | workload 级胜率、bootstrap CI 和建议 |
| `roi_v2_environment.json` | Thor、PyTorch、CUDA、TF32、实验参数 |
| `roi_v2_e2e_by_arm.png` | 各 arm E2E 平均耗时 |
| `delta_adjacency_*.png` | 各 adjacency arm 相对 ROI-Batch 的配对 E2E 差值 |
| `roi_v2_e2e_vs_tile_area.png` | E2E 与 tile 输入面积关系 |
| `run.log` | 完整终端日志（由 tee 保存） |

## 11. 从 Thor 下载结果到 Windows

Thor 上打包：

```bash
cd /home/uestc/uniad
 tar -czf /tmp/roi_dispatch_v2_results.tar.gz work_dirs/roi_dispatch_v2_thor_tf32
```

Windows PowerShell 下载：

```powershell
scp uestc@192.168.112.133:/tmp/roi_dispatch_v2_results.tar.gz "C:\Users\12148\Downloads\"
```

将压缩包上传到 ChatGPT 对话。不要上传 nuScenes 原始数据或 checkpoint。

## 12. 重要限制

- 必须在 Thor 上运行；本地 CPU 单元测试不能替代 GPU/E2E 验证。
- `roi_block_gpu_stream_ms`、各组件 CUDA Event 和 CPU enqueue 时间可能重叠，不可直接相加。
- 最终选择以 `e2e_forward_ms` 为主；ROI-block GPU Stream 和组件时延用于解释原因。
- 由于 ROI 是合成的，结论只适用于本轮覆盖的 ROI 数量、面积、布局和长宽比分布。真实 ROI 分布需要后续从实际模型投影/激活区域采样，再做一次验证。
- 实验脚本不会自动修改 UniAD 正式模型代码，也不会自动部署动态 dispatcher。
