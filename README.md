# Qwen3.5 VLM 实验脚本

脚本覆盖数据构建、样本导出、模型推理、质量评估和性能采集。以下命令从仓库根目录执行，`/model` 表示本机或容器内的模型目录，使用时替换为实际路径。

## 脚本索引

| 脚本 | 功能 | 配置入口 |
|---|---|---|
| [s01_memory_probe.py](src/s01_memory_probe.py) | 权重加载显存与参数布局检查 | 脚本中的 MODEL、OUT、CUDA_VISIBLE_DEVICES |
| [s01_processor_probe.py](src/s01_processor_probe.py) | CPU 输入长度与视觉 token 估算 | CLI |
| [s01_single_reference_probe.py](src/s01_single_reference_probe.py) | 单张参考图重复推理 | CLI |
| [s02_build_baseline_dataset.py](src/s02_build_baseline_dataset.py) | 构建 train/val/test 数据清单 | BUILD_SETTINGS 与 CLI |
| [s02_export_samples.py](src/s02_export_samples.py) | 导出固定样本及图像 | [同名 JSON](src/s02_export_samples.json) |
| [s02_sample_eval.py](src/s02_sample_eval.py) | 吞吐、阶段计时、显存及 operator trace | [同名 JSON](src/s02_sample_eval.json) |
| [s02_compare_vision_backends.py](src/s02_compare_vision_backends.py) | 生成并运行视觉后端成对配置 | 样本评估 JSON 与 CLI |
| [s02_profile_ncu.sh](src/s02_profile_ncu.sh) | 配置并启动 NCU 采集 | 环境变量与脚本参数 |
| [s02_profile_workload.py](src/s02_profile_workload.py) | 执行带 NVTX 标记的固定生成负载 | CLI，通常由 NCU 脚本调用 |
| [s02_evaluate_baseline.py](src/s02_evaluate_baseline.py) | 验证集或测试集质量评估 | CLI |
| [s02_workload.py](src/s02_workload.py) | 样本加载、输入处理和生成公共函数 | 供其他脚本导入 |
| [experiment_profiler.py](src/experiment_profiler.py) | TimingProfiler、MemoryProfiler、OperatorProfiler | 供其他脚本导入 |

数据通常放在 `data/`，完整数据清单放在 `datasets/`，导出的性能样本放在 `samples/`，运行结果放在 `runs/`。图像与标注复核工具见 [visualizer 使用说明](visualizer/README.md)。

## 配置与环境

模型脚本需要可用的 PyTorch、Transformers、Accelerate、Pillow 和本地模型文件；GPU 推理需要 CUDA 环境。数据构建需要 Python 3.10+ 和 Pillow。运行前可单独检查环境。

```bash
# 检查 Python 依赖是否存在冲突。
python -m pip check

# 检查模型接口、框架版本及 CUDA 是否可用。
python -c 'import torch, transformers; from transformers import Qwen3_5ForConditionalGeneration; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
```

`s02_export_samples.py` 和 `s02_sample_eval.py` 默认读取脚本目录下的同名 JSON，支持 `--config` 指定其他文件。JSON 内相对路径以该 JSON 所在目录为基准，CLI 示例中的相对路径以当前工作目录为基准。

模型路径、GPU、batch 和输出目录应按运行机器配置。新运行使用新的输出目录，正式评估恢复已有结果使用 `--resume`。

## s01_memory_probe.py

测量 BF16 权重加载前后的显存、加载耗时和参数布局，并保存显存快照。运行前修改脚本顶部的 `MODEL`、`OUT` 和 `CUDA_VISIBLE_DEVICES`。该脚本在顶层加载模型，直接运行即可。

```bash
python src/s01_memory_probe.py
```

输出到带时间戳的子目录，包含 `load_metrics.json` 和显存快照 `load_memory.pickle`。

## s01_processor_probe.py

在 CPU 上检查单图 processor 结果，估算视觉 token、输入长度和生成预留。`--sample` 接受含 `image`、`expected_detection` 的单样本 JSON，也接受仅含一个样本的 JSON 数组。相对图像路径基于样本 JSON 所在目录。

```bash
python src/s01_processor_probe.py \
  --model /model --sample data/reference.json --out-dir runs/processor_probe
```

输出 `processor_<时间戳>.json`。

## s01_single_reference_probe.py

对参考图重复生成原图像素坐标的眼部框及状态，记录加载、推理时间和显存。`--repeat` 控制首次推理后的重复次数。

```bash
python src/s01_single_reference_probe.py \
  --model /model --image data/reference.jpg --gpu 0 \
  --out runs/reference_v1 --repeat 3 \
  --context-budget 5600 --max-new-tokens 256
```

输出 `preparation.json`、`prompt.txt`、加载与逐次推理指标，以及 `summary.json`。

## s02_build_baseline_dataset.py

读取整图及同名标注，按会话生成 train/val/test 清单。先核对脚本内 `BUILD_SETTINGS` 的目录及批次映射。`--out` 要求新目录。

```bash
python src/s02_build_baseline_dataset.py \
  --data-root data --out datasets/baseline_v1
```

可用 `--observed-list` 传入已经查看的诊断图像清单，每行填写相对于 `--data-root` 的路径，将这些图片排除在最终测试集之外。

输出 `dataset.json`、各 split 的 manifest 和排除/错误报告。manifest 使用绝对图像路径，迁移数据目录后应在目标环境重建。标注字段 `gt_pixel` 为像素 xyxy，`target_norm1000` 为 0～1000 整数 xyxy。

## s02_export_samples.py

在同名 JSON 中填写 `dataset`、`out`、`sample_count`、`sample_seed`。默认从 train 无放回抽取 32 张，样本不足时报错。

```bash
python src/s02_export_samples.py
# 或指定配置
python src/s02_export_samples.py --config src/s02_export_samples.json
```

输出目录必须不存在，迁移时整体复制。

```text
samples/perf_32/
├── samples.jsonl
├── sample_info.json
└── images/
    ├── 0001.jpg
    └── ...
```

`samples.jsonl` 每行保存 sample_id、相对 image 路径、尺寸和处理后的标注。`sample_info.json` 保存抽样参数与来源信息。图片按原始字节复制；性能脚本读取清单中的 sample_id 和 image，图像路径相对于清单目录解析。

## s02_sample_eval.py

在同名 JSON 中配置 `model`、`sample_manifest`、`out`、`mode`、`batch_sizes`、`gpus` 和预热/生成参数。默认配置扫描 B1/2/4/8/16/32，运行前按显存容量调整。

```bash
python src/s02_sample_eval.py
python src/s02_sample_eval.py --config src/s02_sample_eval.json
```

| mode | 功能 | 专用字段 |
|---|---|---|
| `benchmark` | 完整样本清单的吞吐、基础阶段耗时和显存峰值 | `repeats` |
| `timing` | 读图、processor、H2D、generate、解码的计时 | `capture_batches` |
| `memory` | 各阶段 allocated/reserved、峰值和增量 | `capture_batches` |
| `operator` | 算子统计与 CPU/GPU trace | `capture_batches` |

切换到诊断模式时移除 `repeats`，填写正整数 `capture_batches`。`warmup_batches` 控制预热，`context_budget` 和 `max_new_tokens` 控制输入及生成预算。`attn_implementation` 接受后端名称，也可分别配置子模型。

```json
{"vision_config": "sdpa", "text_config": "sdpa"}
```

`gpus` 使用 GPU ID/UUID 字符串列表。多卡仅用于 `benchmark`，还需设置属于 `batch_sizes` 的 `dp_batch_size`。每卡使用独立进程及模型副本，分片处理样本。

输出配置、解析后的样本清单、`summary.json` 和逐档结果。`operator` 另保存 `trace.json` 与 `operators.json`，包含 vision、language_prefill、language_decode 及 logits 阶段标记。时间单位为微秒。普通吞吐使用 benchmark 结果，profiler 输出用于执行过程分析。OOM、超预算或运行错误会停止任务。

## s02_compare_vision_backends.py

生成成对配置并依次调用 `s02_sample_eval.py`。脚本将 text 后端固定为 SDPA，vision 后端分别设为 SDPA、FlashAttention-2，其他负载参数继承基础 JSON。

运行前单独检查 FlashAttention 依赖和变长接口。环境需提前安装与当前 PyTorch/CUDA 兼容的 flash-attn。

```bash
python -c 'import torch, transformers, flash_attn; from flash_attn import flash_attn_varlen_func; print(torch.__version__, torch.version.cuda, transformers.__version__, flash_attn.__version__)'

python src/s02_compare_vision_backends.py \
  --model /model --sample-manifest samples/perf_32/samples.jsonl \
  --out runs/vision_backend_v1
```

| 参数 | 用途及默认行为 |
|---|---|
| `--config` | 基础配置，默认 `src/s02_sample_eval.json` |
| `--model`、`--sample-manifest` | 覆盖模型及样本路径 |
| `--gpu`、`--batch-sizes` | 默认单卡 0，batch 为 1、8 |
| `--modes` | 默认依次 operator、timing、benchmark，也可选 memory |
| `--capture-batches`、`--repeats` | 默认诊断采 1 批，benchmark 重复 3 次 |
| `--prepare-only` | 只生成配置并打印命令 |
| `--reverse` | 每个模式先运行 FA2，再运行 SDPA |

`--out` 必须为新目录，生成的配置位于 `configs/`，结果位于 `sdpa_operator/`、`fa2_operator/` 等子目录。任一子进程失败即停止并保留结果。单项重跑时修改对应 JSON 的 `out`，再使用 `s02_sample_eval.py --config` 执行。

## s02_profile_ncu.sh 与 s02_profile_workload.py

NCU 脚本配置采集条件，调用带 NVTX 标记的 Python 负载。运行环境需要 Nsight Compute 及 GPU 性能计数器访问权限。

```bash
# 确认 NCU 可执行文件和版本。
ncu --version

# 预览定向 basic 命令，确认路径与过滤条件。
MODEL_DIR=/model SAMPLE_MANIFEST=samples/perf_32/samples.jsonl \
  RESULTS_DIR=runs/ncu BATCH_SIZE=1 \
  bash src/s02_profile_ncu.sh --print-command
```

移除 `--print-command` 后开始采集。

| 环境变量 | 用途及默认值 |
|---|---|
| `MODEL_DIR` | 模型目录，默认 `/model` |
| `SAMPLE_MANIFEST` | 清单，默认 `/samples/samples.jsonl` |
| `RESULTS_DIR` | 输出根目录，默认 `/out`；本地可设为 `runs/ncu` |
| `BATCH_SIZE` | 默认 1 |
| `NCU_SCOPE` | 默认 `target`；`measure` 用于整个 measure 范围的 basic 采集 |
| `NCU_SET` | 默认 `basic`，定向采集支持 `full` |
| `PHASE` | 默认 `language_prefill`；支持 vision、language_decode、prefill_logits、decode_logits |
| `KERNEL` | kernel 名正则；默认值在脚本顶部，使用前核对目标名称 |
| `LAUNCH_SKIP`、`LAUNCH_COUNT` | basic 默认 0、4；full 必须显式设置 |

例如，已经核对目标是当前阶段及名称过滤范围内的第 3 次匹配时，使用 `LAUNCH_SKIP=2`、`LAUNCH_COUNT=1`。

```bash
MODEL_DIR=/model SAMPLE_MANIFEST=samples/perf_32/samples.jsonl \
  RESULTS_DIR=runs/ncu BATCH_SIZE=8 NCU_SET=full \
  LAUNCH_SKIP=2 LAUNCH_COUNT=1 bash src/s02_profile_ncu.sh
```

GPU、预热、采集批数、生成预算、CPU 线程数和 attention 后端配置在 shell 的 `workload` 数组中。每轮创建唯一目录，保存 `command.sh`、`report.ncu-rep` 和 `workload/workload.json`。

Python 负载也可以直接运行，以下参数均需提供。它保存环境、样本、输入形状和生成结果，供外部 profiler 采集。

```bash
python src/s02_profile_workload.py \
  --model /model --sample-manifest samples/perf_32/samples.jsonl \
  --out runs/profile_workload_v1 --batch-size 1 --gpu 0 \
  --warmup-batches 2 --capture-batches 1 \
  --context-budget 5600 --max-new-tokens 256 \
  --cpu-threads-per-worker 4 --attn-implementation sdpa
```

清单需满足 `batch_size × capture_batches` 的完整批次要求。NVTX 提供 measure、vision、language_prefill、language_decode、prefill_logits 和 decode_logits 范围。NCU 与 torch.profiler 分开运行。

## s02_evaluate_baseline.py

读取完整数据集逐图推理，评估解析、定位、眼部状态及端到端指标。显式传入模型路径。

```bash
# CPU 检查数据与评估输入。
python src/s02_evaluate_baseline.py --dataset datasets/baseline_v1 \
  --out runs/baseline_preflight_v1 --preflight-only

# 默认评估 val。
python src/s02_evaluate_baseline.py --dataset datasets/baseline_v1 \
  --model /model --gpu 0 --out runs/baseline_val_v1 \
  --context-budget 5600 --max-new-tokens 256
```

`--resume` 使用相同配置恢复已有输出，`--save-overlays` 保存 GT/预测叠框，恢复时保持该选项一致。测试集使用 `--split test --final-test`。

输出 `config.json`、`cases/`、`summary.json`、`by_batch.json`。GT 使用原图像素 xyxy，模型输出按 Norm1000 xyxy 还原为像素评分。失败样本计入分母；四类状态覆盖不全时 macro-F1 为 null，`evaluation_complete` 表示清单处理完整。
