#!/usr/bin/env bash
# 多卡并行：与 scripts/eval_fastv/textvqa_parallel.sh 相同结构，入口为 model_vqa_loader_diff。
#
#   NUM_GPUS=4 bash scripts/eval_diff/textvqa_parallel.sh
#
# 可选：MODEL_PATH, NUM_GPUS, DISABLE_PRUMERGE, MERGED_JSONL, PART_DIR, FASTV_DIFF_LAMBDA_TT

set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

NUM_GPUS="${NUM_GPUS:-4}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MERGED_JSONL="${MERGED_JSONL:-./playground/data/eval/textvqa/answers/llava-v1.5-7b-fastv-diff.jsonl}"
PART_DIR="${PART_DIR:-./playground/data/eval/textvqa/answers/_fastv_diff_parallel_parts}"
export DISABLE_PRUMERGE=1
LAMBDA_TT="${FASTV_DIFF_LAMBDA_TT:-1.0}"

mkdir -p "$PART_DIR"
rm -f "$PART_DIR"/part_*.jsonl

pids=()

_parallel_kill_workers() {
  local p
  for p in "${pids[@]:-}"; do
    if kill -0 "$p" 2>/dev/null; then
      kill -TERM "$p" 2>/dev/null || true
    fi
  done
  for p in "${pids[@]:-}"; do
    wait "$p" 2>/dev/null || true
  done
}

trap '_parallel_kill_workers; echo "[parallel] SIGINT，已终止 worker。"; exit 130' INT
trap '_parallel_kill_workers; echo "[parallel] SIGTERM"; exit 143' TERM

for i in $(seq 0 $((NUM_GPUS - 1))); do
  echo "[parallel] GPU $i  chunk ${i}/${NUM_GPUS}"
  CUDA_VISIBLE_DEVICES=$i python -m llava.eval.model_vqa_loader_diff \
    --model-path "${MODEL_PATH}" \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file "${PART_DIR}/part_${i}.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --fastv-k 2 \
    --fastv-r 0.5 \
    --fastv-sys-len 35 \
    --fastv-image-len 576 \
    --fastv-diff-lambda-tt "${LAMBDA_TT}" \
    --num-chunks "${NUM_GPUS}" \
    --chunk-idx "${i}" &
  pids+=($!)
done

wait_fail=0
set +e
for pid in "${pids[@]}"; do
  wait "$pid"
  s=$?
  if [[ $s -ne 0 ]]; then
    wait_fail=$s
    echo "[parallel] worker pid=${pid} 退出码=${s}"
    break
  fi
done
set -e

if [[ $wait_fail -ne 0 ]]; then
  _parallel_kill_workers
  exit "$wait_fail"
fi

trap - INT TERM

echo "[parallel] 合并到 ${MERGED_JSONL}"
: > "${MERGED_JSONL}"
for i in $(seq 0 $((NUM_GPUS - 1))); do
  cat "${PART_DIR}/part_${i}.jsonl" >> "${MERGED_JSONL}"
done

python -m llava.eval.eval_textvqa \
  --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
  --result-file "${MERGED_JSONL}"

echo "[parallel] 完成。"
