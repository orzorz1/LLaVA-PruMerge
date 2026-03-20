#!/usr/bin/env bash
# 多卡并行推理（每卡一个进程 + 数据分片），不改 Python。
# 依赖 model_vqa_loader_fastv 的 --num-chunks / --chunk-idx。
#
# 用法（在项目根目录）：
#   NUM_GPUS=4 bash scripts/eval_fastv/textvqa_parallel.sh
# 指定前 N 张卡：0..N-1，每卡跑一块数据，最后合并 jsonl 再 eval。
#
# Ctrl+C / SIGTERM：会向本脚本启动的所有 python 子进程发 SIGTERM，避免「停一个、其余仍跑」。
#
# 可选环境变量：
#   MODEL_PATH, NUM_GPUS, DISABLE_PRUMERGE, MERGED_JSONL, PART_DIR

set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

NUM_GPUS="${NUM_GPUS:-4}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MERGED_JSONL="${MERGED_JSONL:-./playground/data/eval/textvqa/answers/llava-v1.5-7b-fastv.jsonl}"
PART_DIR="${PART_DIR:-./playground/data/eval/textvqa/answers/_fastv_parallel_parts}"

export DISABLE_PRUMERGE="${DISABLE_PRUMERGE:-1}"

mkdir -p "$PART_DIR"
rm -f "$PART_DIR"/part_*.jsonl

pids=()

# 杀掉本脚本拉起的所有 worker（子进程的子进程一般会随 Python 一起退出）
_parallel_kill_workers() {
  local p
  for p in "${pids[@]:-}"; do
    if kill -0 "$p" 2>/dev/null; then
      kill -TERM "$p" 2>/dev/null || true
    fi
  done
  # 回收僵尸，不强制要求 0 退出码
  for p in "${pids[@]:-}"; do
    wait "$p" 2>/dev/null || true
  done
}

# Ctrl+C、kill 脚本时：先清 worker，再退出（避免后台 python 继续占 GPU）
trap '_parallel_kill_workers; echo "[parallel] 已发送 SIGTERM 给所有 worker 并退出。"; exit 130' INT
trap '_parallel_kill_workers; echo "[parallel] 已发送 SIGTERM 给所有 worker（SIGTERM）"; exit 143' TERM

for i in $(seq 0 $((NUM_GPUS - 1))); do
  echo "[parallel] GPU $i  chunk ${i}/${NUM_GPUS}"
  CUDA_VISIBLE_DEVICES=$i python -m llava.eval.model_vqa_loader_fastv \
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
    --num-chunks "${NUM_GPUS}" \
    --chunk-idx "${i}" &
  pids+=($!)
done

# 任一 worker 非 0 退出：终止其余仍在跑的 worker（set -e 下原先会漏掉清理）
wait_fail=0
set +e
for pid in "${pids[@]}"; do
  wait "$pid"
  s=$?
  if [[ $s -ne 0 ]]; then
    wait_fail=$s
    echo "[parallel] worker pid=${pid} 退出码=${s}，正在终止其余 worker…"
    break
  fi
done
set -e

if [[ $wait_fail -ne 0 ]]; then
  _parallel_kill_workers
  exit "$wait_fail"
fi

# 子进程已全部正常结束，勿在合并阶段再响应 INT 时误杀（已无子进程）；取消 trap 以免合并时误触
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
