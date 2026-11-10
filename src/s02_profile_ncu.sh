#!/usr/bin/env bash
# 在实验容器内执行；工作负载参数和 NCU 参数统一在此设置。
set -euo pipefail
src_dir=$(cd "$(dirname "$0")" && pwd)
batch=${BATCH_SIZE:-1}
set_name=${NCU_SET:-basic}
scope=${NCU_SCOPE:-target}
phase=${PHASE:-language_prefill}
kernel=${KERNEL:-'regex:^ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn$'}
results=${RESULTS_DIR:-/out}
if [[ $# != 0 && ( $# != 1 || $1 != --print-command ) ]]; then
  echo "用法：$0 [--print-command]" >&2; exit 2
fi
case "$scope:$set_name" in
  measure:basic)
    filters=(--nvtx-include 'measure/') ;;
  target:basic|target:full)
    case "$phase" in vision|language_prefill|language_decode|prefill_logits|decode_logits) ;; *) echo "未知阶段：$phase" >&2; exit 2 ;; esac
    if [[ $set_name == full ]]; then
      : "${LAUNCH_SKIP:?full 需指定已核对的 LAUNCH_SKIP}" "${LAUNCH_COUNT:?full 需指定 LAUNCH_COUNT}"
    fi
    skip=${LAUNCH_SKIP:-0}
    count=${LAUNCH_COUNT:-4}
    [[ $skip =~ ^[0-9]+$ && $count =~ ^[1-9][0-9]*$ ]] || { echo "skip/count 无效" >&2; exit 2; }
    filters=(--nvtx-include "measure/$phase" --kernel-name "$kernel" --launch-skip "$skip" --launch-count "$count") ;;
  *) echo "支持 target:basic、target:full 或 measure:basic" >&2; exit 2 ;;
esac

if [[ ${1:-} == --print-command ]]; then
  run="$results/preview"
else
  mkdir -p "$results"
  run=$(mktemp -d "$results/ncu-b${batch}-${set_name}.XXXXXX")
fi

# 固定输入、生成和预热条件；在 B1/B8 对照中只改变 batch。
workload=(python "$src_dir/s02_profile_workload.py"
  --model "${MODEL_DIR:-/model}" --sample-manifest "${SAMPLE_MANIFEST:-/samples/samples.jsonl}"
  --batch-size "$batch" --gpu 0 --warmup-batches 2 --capture-batches 1
  --context-budget 5600 --max-new-tokens 256 --cpu-threads-per-worker 4
  --attn-implementation sdpa --out "$run/workload")

command=(ncu --nvtx --kernel-name-base demangled --set "$set_name"
  "${filters[@]}" -o "$run/report" "${workload[@]}")
printf '%q ' "${command[@]}"; printf '\n'
if [[ ${1:-} == --print-command ]]; then exit 0; fi
{ printf '%q ' "${command[@]}"; printf '\n'; } > "$run/command.sh"
exec "${command[@]}"
