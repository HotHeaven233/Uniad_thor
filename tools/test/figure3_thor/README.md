# Thor Figure 3 — journal extension, Baseline workload characterization

安装路径：`/home/uestc/uniad/tools/test/figure3_thor/`。这是扩刊方案的新 Figure 3，不是原会议论文的 Figure 3。

## 范围和已知限制

- `full` 是真实原始 UniAD 2.0 TF32 E2E（不改变原网络计算路径），运行时仅包装方法计时；无需改写 `projects/`。全部测量执行真实 nuScenes `model(return_loss=False,rescale=True,**data)`。
- 当前 Thor 原始 QC-BEV Scheduler 及原论文整套稀疏执行机制还未完成移植；此包不声称完成该移植。`original-qc` 是为完成移植后的真实 QC-BEV 留出的采集入口，必须指定实际运行的 scheduler 方法；若未调用则终止。可选 adapter 文件定义 `install(model,cfg)`，负责真实代码接入，不提供伪造调度器。用户须检查实际执行路径及正确性。
- `area` 是控制变量实验：使用真实图像中固定中心裁剪，单相机 B=1，64 对齐（FPN 最大 stride 64），计算六个相机后回填全尺寸特征图。完整 E2E 仍执行，但不代表实际 QC-BEV Query 投影，不用于准确率或原论文 QC-BEV 基线。此实验只用于观测小输入面积的 CNN stream span；不能仅凭平台期推断全部时间是 launch overhead。
- 采集脚本用 Thor repo 的 `simple_test_track`, `get_bevs`, `extract_img_feat`, `pts_bbox_head.get_bev_features`, `seg_head.forward_test`, `motion_head.forward_test`, `occ_head.forward_test`, `planning_head.forward_test`。结构不符或应运行的模块未执行即报错。全量 detail 校验六层 TSA/SCA/FFN/norm 调用次数。修改版本后需重新核对方法名。

## 运行

```bash
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
P=tools/test/figure3_thor
python -m py_compile "$P/"*.py
(cd "$P" && python -m unittest -v test_figure3_plot.py)
bash "$P/run_figure3_thor.sh" smoke
bash "$P/run_figure3_thor.sh" full
bash "$P/run_figure3_thor.sh" trace
R=work_dirs/figure3_thor_full
python "$P/plot_figure3_thor.py" --summary "$R/summary" --detail "$R/detail" --area "$R/area" --plain "$R/plain" --trace-sqlite work_dirs/figure3_thor_trace/nsys.sqlite --out-dir "$R/report"
```

Smoke/full/trace 分别使用独立输出目录。已有 `frames.csv` 时拒绝覆盖；失败后保存原目录，使用 `bash "$P/run_figure3_thor.sh" smoke work_dirs/figure3_thor_smoke_retry`。Trace 需 Thor 上可运行 Nsight Systems CLI (`nsys --version`)，具体选项和 SQLite schema 可能因版本不同而需适配。如果无 nsys，绘图仍可生成其他面板并明确标示 b 缺失。Nsight 只采集一帧，不把被 CUPTI 插桩的时间混入正式 summary。

## 数据与计时协议

summary 默认 12 scene × 8 measured frames × 2 repeats（场景长度足够时 192）；每段前面使用真实连续 3 帧 warmup。plain 无模块插桩，使用 summary 的同一 manifest，比较 E2E 插桩扰动。detail 6×4×2（48）；area 3 scenes × 8 shapes × 2 repeats（48），各 shape 前重置历史并用同样真实前序帧 warmup，6 相机每帧采集。trace 1 frame。所有模式 FP32 tensor、TF32 enabled，无 AMP/FP16。采集前固定功耗/时钟/散热设置；程序只记录 nvpmodel、jetson_clocks、环境，不擅自改变。无 module-internal synchronize；每帧开始前同步，forward 完成后同步。模型中原本存在的同步调用保留。连续场景内时序缓存保留；scene/repeat 边界重置。

- `frames.csv`: 完整 Forward wall (含 MMDataParallel scatter 和最终 drain，不含 dataset load)、host return 前时间、load、memory。
- `scopes.csv`: 模块嵌套 ID、父 ID、host timestamps、CUDA event relative stream timestamps；无 profiler 的 plain 目录 scopes.csv 为空。
- `report/table_iv_module_stats.csv`: 分模块 host / CUDA stream exclusive span 和 E2E mean/median/P90/P95/P99。Tracking 已排除嵌套的 BEV 构造部分；scheduler absent in Full，表中的 0 表示未执行，不是 QC-BEV scheduler 0ms。`outside_gpu_ms` 仅代表被记录 scope 的类别，不是未记录 GPU 工作的余量。GPU span 包含 GPU 空闲间隔，host scope 可能含前序 GPU wait，两者不相加。Host exclusive + outside + final sync 为观测 wall-clock 区间分解，不是精确硬件归因；最终论文需结合 Nsight。
- `figure3_thor.pdf/png`: (a) host wall partition, (b) real launch/kernel timeline (requires trace), (c) Encoder TSA/SCA/FFN/Norm/point-sampling/exclusive residual, (d) measured crop pixels vs ResNet+FPN stream spans, median and P10–P90.
- `host_vs_stream.pdf/png`: 单独并列 host vs stream，不能累加。`encoder_operator_frame.csv`, `area_camera_raw.csv`, `frame_exclusive_spans.csv`, `report.json`, trace `nsys_api_stats.csv`, `nsys_trace_summary.json`。
- `environment.json`, `manifest.json`, resolved config, `COMPLETE`: 复现信息。原始日志保留；不完整运行不出正式报告。

## 将来接入真实 QC-BEV 后

在已验证的原始 QC-BEV on Thor 配置上（必须已移植原论文代码，不是 V3 synthetic ROI），保持相同 checkpoint/TF32/manifest/repeats，例：

```bash
python -u tools/test/figure3_thor/profile_figure3_thor.py "$QC_CONFIG" ckpts/uniad_base_e2e.pth \
 --mode original-qc --scheduler-method "$ACTUAL_SCHEDULER_METHOD" \
 --profile summary --manifest work_dirs/figure3_thor_full/summary/manifest.json \
 --repeats 2 --out-dir work_dirs/figure3_thor_qc/summary
```

`ACTUAL_SCHEDULER_METHOD` 必须是实际 model 上可访问的点分路径，例如由最终 adapter 显式暴露的 method（不猜现有名字）。如果 scheduler 是模块级函数，需通过真实 adapter 暴露实际调用路径而不是挂空函数；optional `--adapter /path/to/real_adapter.py`。然后 `--summary .../full/summary .../qc/summary` 生成配对对比。跨版本测量必须冻结 commit/config/weight，不能把 area 模式或 full 模式当 QC 基线。完整论文 Figure 3 的 QC 对照、原论文 Backbone 和 Planner 重新拟合仍待实施。

## 本地验证

语法编译、bash -n、CPU-only interval union / nesting / multiple-stream rejection / mock CSV & SQLite report pipeline。没有 Thor GPU，因此未执行 CUDA/E2E 真实验证。模拟数据仅用于单元测试，不打包为实验结果。
