# Qwen3.5 VLM 实验

本仓库维护 VLM 数据构建、精度评估、推理与性能诊断。当前正式实验使用 4B，0.8B 用于本地环境验收，9B 结果作为历史记录。

本地原图与标注位于 `data/`，完整数据清单位于 `datasets/baseline_v1/`，固定性能样本位于 `samples/perf_32/`。这三类数据由本地提供；在其他机器上准备原图后，从仓库根目录执行：

```bash
python3 src/s02_build_baseline_dataset.py --data-root data --out datasets/baseline_v1
```

已有本地清单可直接使用。构建入口要求新输出目录。性能和推理实验另需按运行机器配置模型、GPU 及输出目录。标注复核见 [visualizer 使用说明](visualizer/README.md)。

## 文件与启动约定

| 入口 | 实验用途与输入 |
|---|---|
| [s01_memory_probe.py](src/s01_memory_probe.py) | 记录 BF16 权重加载前后显存、参数布局和显存快照。脚本顶层即加载模型；运行前检查文件内的模型路径、输出根目录和可见 GPU。 |
| [s01_processor_probe.py](src/s01_processor_probe.py) | 在 CPU 上估算整图视觉 token、输入长度和生成预留。传入 `--model`、`--sample`、`--out-dir`；样本为含 `image` 和 `expected_detection` 的单样本 JSON。 |
| [s01_single_reference_probe.py](src/s01_single_reference_probe.py) | 对一张参考图重复生成，检查原图像素坐标输出并记录时间与显存。复用时显式传入 `--model`、`--image`、`--gpu`、`--out`。 |

| 文件 | 用途 | 配置 |
|---|---|---|
| [s02_build_baseline_dataset.py](src/s02_build_baseline_dataset.py) | 构建数据清单 | 脚本内BUILD_SETTINGS与CLI |
| [s02_export_samples.py](src/s02_export_samples.py) | 从完整训练集导出固定清单和原图 | [同名JSON](src/s02_export_samples.json) |
| [s02_sample_eval.py](src/s02_sample_eval.py) | 样本吞吐、batch/DP与阶段/算子测量 | [同名JSON](src/s02_sample_eval.json) |
| [s02_profile_workload.py](src/s02_profile_workload.py) | 为Nsight提供固定生成负载 | [NCU 采集脚本](src/profile_ncu.sh) |
| [s02_evaluate_baseline.py](src/s02_evaluate_baseline.py) | 完整验证集质量评估 | CLI |
| [s02_workload.py](src/s02_workload.py) | 共享输入、模型加载与生成语义 | 供样本效率与Nsight入口调用 |
| [experiment_profiler.py](src/experiment_profiler.py) | 通用计时、显存和算子采集 | 库模块 |

导出样本和样本评估这两个JSON入口默认读取脚本自身目录中的同名`.json`，不依赖启动时的工作目录。也可用`--config /path/to/config.json`覆盖；JSON内部的相对路径始终基于该JSON目录。默认的是配置文件位置，实验参数仍要求显式填写。

示例配置路径相对于src目录。导出脚本读取完整dataset；样本评估默认读取`../samples/perf_32/samples.jsonl`，NCU 在容器中默认读取同一固定清单的挂载路径 `/samples/samples.jsonl`，与benchmark输出目录独立。填写当前机器实际的4B模型目录和GPU，每轮使用新的out。以下命令从项目根目录执行。



## s02_build_baseline_dataset.py：构建数据

读取整图及同名标注，按完整会话目录划分train/val/test，生成dataset.json、三个manifest和排除/错误报告。先核对BUILD_SETTINGS中的批次映射和真实目录名。

```bash
python src/s02_build_baseline_dataset.py \
  --data-root /workspace/dms-eye-status-turbo/extract_data \
  --out datasets/baseline_v1
```

`--observed-list`可传入已查看诊断图片的相对路径清单，防止其进入最终测试集。manifest保存图片绝对路径；迁移图片目录后在实际环境重建。构建需要Python 3.10+和Pillow。

## s02_export_samples.py：导出固定性能样本

在完整dataset可访问的机器上，填写同名JSON中的dataset和out，默认从train均匀随机无放回抽取32张，seed为20261008。按sample_id排序后抽样，保留抽样顺序；样本不足32张时明确报错。修改数量需显式调整sample_count。

```bash
python src/s02_export_samples.py
```

导出目录如下，迁移时整体复制：

```text
samples/perf_32/
├── samples.jsonl
├── sample_info.json
└── images/
    ├── 0001.jpg
    └── ...
```

原图按字节复制，不重新编码。样本数据与溯源统计分为两个文件，用sample_id关联：

| 文件 | 内容与用途 |
|---|---|
| samples.jsonl | sample_id、相对image路径、width、height、gt_pixel、target_norm1000；图像与处理后的标注放在同一行，清单无需split |
| sample_info.json | 抽样seed、数量、source_dataset、source_split，以及samples列表中的原图/原标注路径、已有SHA256、批次和场景等溯源信息 |

标注沿用完整dataset中的原字段和坐标含义：gt_pixel为原图像素xyxy，target_norm1000为0～1000整数xyxy，导出时不重新计算。性能入口只使用sample_id和image，其余字段保留供训练或输出检查使用。溯源信息独立保存，不参与模型输入或评分；正式质量评估仍使用完整dataset。

source_dataset和逐样本原路径仅用于追溯；迁移后运行只依赖samples.jsonl及images。输出目录已存在时拒绝覆盖；导出失败时清理本次临时目录。修改前已导出的目录保持原样，需要在新的out中重新导出才能得到这一布局。

两个性能入口统一用sample_manifest指定导出的清单，图片路径始终相对清单目录解析，与启动目录无关。入口检查唯一ID、唯一图像路径和文件存在；训练集来源在导出时确认，运行时不依赖split字段。旧运行结果里的绝对路径清单不再作为新输入。正式评估仍使用完整dataset。

## s02_sample_eval.py：效率测量

默认JSON为benchmark：读取导出的32张固定训练样本，扫描B1/2/4/8/16/32。不读取GT评分，不替代正式baseline。

```bash
python src/s02_sample_eval.py
```

通过同名JSON的mode选择功能，每次修改out：

| mode | 测什么 | 条件字段 |
|---|---|---|
| benchmark | 完整清单的吞吐、五阶段基础耗时及整轮显存峰值 | repeats |
| timing | 读图、processor、H2D、generate、输出解码的CPU/CUDA事件区间 | capture_batches |
| memory | 五阶段allocated/reserved入口、结束、峰值与增量 | capture_batches |
| operator | 流程和vision/prefill/decode区间、算子统计与trace | capture_batches |

切换到timing/memory/operator时移除repeats，增加正整数capture_batches，例如3；sample_manifest继续指向同一导出清单。batch_sizes控制扫描档位，warmup_batches控制预热；dataset、sample_count和sample_seed只属于导出配置，性能入口不再抽样。

gpus为GPU ID/UUID字符串列表。多卡仅benchmark支持，增加dp_batch_size并要求该档位属于batch_sizes；每卡独立进程和完整模型，样本无重复分片，以全部卡完成的全局窗口比较吞吐。

输出config.json、resolved_samples.json、sample_info.json、summary.json及逐档结果；resolved_samples.json保存本轮解析后的绝对路径，仅作为运行快照；operator另存profile_b*/rank0/trace.json与operators.json。计时单位为微秒，批耗时除以图片数是摊销耗时。profiler数据用于归因，普通benchmark用于比较收益；OOM、超预算或运行错误停止，不静默缩小batch。

operator模式统一承担算子次数、CPU/CUDA耗时、内存摘要与trace采集，不再设置独立的Python attention接口核验入口。`operators.json`包含`count`、`cpu_total_us`、`self_cpu_us`、`device_total_us`和`self_device_us`等字段，是整个采集窗口按事件名聚合的统计；嵌套事件的total时间不能直接相加。

trace保留`pipeline/`流程范围和`experiment/vision`、`experiment/language_prefill`、`experiment/language_decode`、`experiment/prefill_logits`、`experiment/decode_logits`模块阶段范围。沿CPU事件的父子关系及时间范围定位阶段，再通过CUDA launch关联ID匹配kernel和输入shape；不能只按CPU/GPU时间重叠判断归属。当前hook标记到阶段，不直接提供每个attention层的路径；只有需要区分同名同shape实例时再补目标模块标记。算子调用次数与kernel次数不要求一一对应。

后续NCU独立运行`s02_profile_workload.py`，不与torch.profiler同时采集。两个运行通过固定负载、阶段、kernel名称、shape和启动配置核对实例，不跨运行复用关联ID或时间戳。已有`execution_inspect_*`历史产物继续保留，新采集使用operator模式。

## s02_profile_workload.py：Nsight负载

读取同一冻结清单，在每个待采集batch完整generate预热后，用NVTX `measure`标记生成及完成同步；内部通过hook标记`vision`、`language_prefill`、`language_decode`及独立的`prefill_logits`/`decode_logits`。语言阶段按进入forward前的cache长度和输入长度判定，不按调用序号猜测；阶段之间不新增同步。输入准备、预热和输出写盘位于范围外。与operator对照时核对固定样本、模型、SDPA后端、输入/生成预算及预热条件，使用独立out；两个入口的采集窗口和预热组织不同，不把各自耗时直接作性能对照。

在 [profile_ncu.sh](src/profile_ncu.sh) 顶部设置模型、输入、预热、生成预算和 NCU 过滤条件。shell 直接启动 NCU，Python 执行一次固定工作负载，记录实际生效参数与输出；每轮输出目录含 report.ncu-rep、command.sh 和 workload/workload.json。源码与输入挂载约定见脚本默认值。

先在 trace 中按阶段统计设备耗时和每图累计耗时，反查热点 kernel 对应的算子与 shape，再按组选择可比实例。4090/4B 的目标 prefill GEMM 在 B8 有 120 次同名调用，其中 64 次对应权重 [2560,9216]。该 shape 首次出现的 grid 在 B1 为 [72,75,1]，在 B8 为 [72,598,1]，block 均为 [64,1,1]。以下为容器内示例：

```bash
# 定向 basic：各核对前 4 个匹配。
BATCH_SIZE=1 bash src/profile_ncu.sh
BATCH_SIZE=8 bash src/profile_ncu.sh

# 两档 basic 分别确认第 1 和第 3 个匹配后才执行 full。
NCU_SET=full BATCH_SIZE=1 LAUNCH_SKIP=0 LAUNCH_COUNT=1 bash src/profile_ncu.sh
NCU_SET=full BATCH_SIZE=8 LAUNCH_SKIP=2 LAUNCH_COUNT=1 bash src/profile_ncu.sh
```

`--print-command` 可预览实际命令。上述序号限于本轮阶段和名称过滤条件；shape/grid 相同且需区分投影时补模块级 NVTX 标记。全窗口盘点使用 `NCU_SCOPE=measure NCU_SET=basic bash src/profile_ncu.sh`。旧的 Python `--ncu` 包装入口和默认 JSON 已由 shell 收敛替代，其他 NCU 参数直接在采集脚本中维护。

- full后核对报告的实际名称、阶段、启动配置和采集数量是否满足所选范围；无匹配、少采、执行失败或实例对应不明时，不作预期目标的硬件结论。
- NVTX标记CPU提交范围。默认kernel replay下，阶段筛选仍生成逐kernel指标，不是阶段整体Roofline；不根据CPU标记跨度计算GPU阶段耗时。
- full内容依云端版本和硬件而定。检查匹配精度/计算路径的Roofline、计算管线、各存储层级吞吐，再结合scheduler、eligible warps、依赖等待及spill判断；缺项按实际支持的section/metric补采。不能仅凭低occupancy或低DRAM带宽确定原因。

输出workload.json，保留实际环境、样本、输入形状、图像网格与生成步数；ncu设置另随报告记录，特别是筛选、replay、缓存和时钟控制。采集会扰动执行，其耗时不作为普通吞吐；优化收益另用benchmark验证。attention的调用组织结合operator trace和模型实现判断；硬件指标不能证明后端不支持batch。

## s02_evaluate_baseline.py：正式质量评估

逐图处理完整验证清单，报告解析、定位、对应侧四类状态和端到端指标；失败保留分母。先做可选CPU检查，再运行真实推理：

```bash
python src/s02_evaluate_baseline.py --dataset datasets/baseline_v1 \
  --out runs/baseline_preflight_v1 --preflight-only
python src/s02_evaluate_baseline.py --dataset datasets/baseline_v1 \
  --model /workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-4B \
  --gpu 0 --out runs/baseline_val_v1 --context-budget 5600 --max-new-tokens 256
```

`--resume`用相同参数恢复，成功与失败case均保留；`--save-overlays`保存GT/预测叠框，恢复时保持该选项一致。最终候选才用`--split test --final-test`。输出config.json、cases/、summary.json、by_batch.json；四类覆盖不全时macro-F1为null，evaluation_complete只代表清单处理完整。

## 共享模块与固定合同

s02_workload.py封装固定prompt、可迁移样本清单校验与路径解析、离线模型加载、图片/processor处理和greedy生成，不拥有测量策略。experiment_profiler.py提供TimingProfiler、MemoryProfiler和OperatorProfiler，可供后续训练循环复用，当前未实现LoRA训练。

- 正式批次：67/68/70/71/75训练、77验证、79测试；完整会话不能拆分，同人不同会话允许跨集合。hard或缺失/非法标注整图排除，构建时检查跨集合相同图片。
- GT按标注ID固定映射：id=2为image_left_eye，id=1为image_right_eye，不按框中心重新排序。schema v4保留像素GT与整数Norm1000 xyxy目标；预测还原一次为浮点像素评分，不补框、不交换左右。
- BF16、离线加载、左padding、非思考greedy、num_beams=1、use_cache=True。示例总预算5600、输出上限256，超限不截断；图像预算沿用processor。样本效率与Nsight负载seed=42，正式评估CLI默认seed=20261003。
- 每次generate不传递上次输出cache。样本效率固定请求SDPA；Nsight负载的attn_implementation显式配置，对照时保持一致。Nsight入口只接受完整采集批次，样本数须达到batch_size×capture_batches；operator入口可能包含尾批，以实际输出的批次和样本数为准。

## 当前状态与依据

同步日期2026-11-10。当前模型统一为4B，本地4090用于NCU学习与小规模优化对照，云端用于优化复验和训练。固定性能样本由独立导出脚本生成，默认32张。9B 表格见历史实验附录。上文 Nsight 示例依据 4090/4B trace，候选匹配位置须用当前设备的定向 basic 确认。历史B16是9B的DP候选。下一步核验实际后端路径并采集代表性vision attention、prefill GEMM硬件指标，DP可独立推进。正式全量baseline仍待运行；profiler与NCU入口分工见上文，本地结构/模拟验证不代表GPU性能或模型质量通过。

知识库实验方案定义方法，实验记录及附录保存历史结果；本说明不复制历史分析。用户后续明确约定优先于旧文档中的GT排序、非法标注处理或schema表述。原文位置见AGENTS.md，历史结果不要求适配当前接口。

参考：[Qwen3.5实现](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。
