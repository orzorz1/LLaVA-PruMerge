#!/usr/bin/env bash
#
# ============================================================================
#  TextVQA 多卡并行评测脚本（原始 LLaVA-v1.5-7B，不加载 PruMerge LoRA）
# ============================================================================
#
# 【原理】
#   将 TextVQA 验证集的问题按 GPU 数量均匀分成 N 个 chunk，
#   每张 GPU 独立推理各自的 chunk（通过 --num-chunks 和 --chunk-idx 控制），
#   所有 GPU 推理完成后，将各 chunk 的结果文件合并为一个完整的 jsonl，
#   最后统一计算评测指标。
#
# 【使用方式】
#   NUM_GPUS=4 bash scripts/eval_llava/textvqa_parallel.sh
#
# 【可选环境变量】
#   MODEL_PATH   : 模型路径，可以是 HuggingFace 名字或本地路径
#                  默认值：liuhaotian/llava-v1.5-7b
#   NUM_GPUS     : 使用的 GPU 数量，默认 4
#   MERGED_JSONL : 最终合并后的答案文件路径
#   PART_DIR     : 各 chunk 的临时输出目录
# ============================================================================

# 遇到任何命令返回非零退出码时，立即终止脚本（防止错误被忽略继续执行）
set -e

# 获取项目根目录的绝对路径（脚本位于 scripts/eval_llava/ 下，所以往上两级就是项目根目录）
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# 切换到项目根目录，确保后续所有相对路径都以项目根目录为基准
cd "$ROOT"

# ----- 配置参数（均可通过同名环境变量覆盖） -----

# 并行 GPU 数量，默认 4 张卡
NUM_GPUS="${NUM_GPUS:-4}"

# 模型路径：HuggingFace 模型名 或 本地绝对路径
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"

# 所有 chunk 合并后的最终答案文件
MERGED_JSONL="${MERGED_JSONL:-./playground/data/eval/textvqa/answers/llava-v1.5-7b-original.jsonl}"

# 各 GPU chunk 推理结果的临时存放目录
PART_DIR="${PART_DIR:-./playground/data/eval/textvqa/answers/_llava_parallel_parts}"

# 禁用 PruMerge token 压缩，使用原始 CLIP 视觉编码（576 tokens）
export DISABLE_PRUMERGE=1

# ----- 准备临时目录 -----

# 创建 chunk 输出目录（如果不存在）
mkdir -p "$PART_DIR"
# 清理上一次运行可能残留的 chunk 文件，避免合并时混入旧数据
rm -f "$PART_DIR"/part_*.jsonl

# ----- 信号处理（优雅退出） -----

# 数组：记录所有后台推理进程的 PID
pids=()

# 辅助函数：终止所有后台 worker 进程
# 先发送 SIGTERM 信号让进程有机会优雅退出，再 wait 等待其完全结束
_parallel_kill_workers() {
  local p
  # 第一轮：向每个还存活的 worker 发送 TERM 信号
  for p in "${pids[@]:-}"; do
    if kill -0 "$p" 2>/dev/null; then    # kill -0 仅检测进程是否存在，不发送信号
      kill -TERM "$p" 2>/dev/null || true # 发送 SIGTERM
    fi
  done
  # 第二轮：等待所有 worker 进程彻底退出，回收资源
  for p in "${pids[@]:-}"; do
    wait "$p" 2>/dev/null || true
  done
}

# 捕获 Ctrl+C（SIGINT）：先杀掉所有 worker，再以退出码 130 退出
trap '_parallel_kill_workers; echo "[parallel] SIGINT，已终止 worker。"; exit 130' INT
# 捕获 SIGTERM：同上，退出码 143
trap '_parallel_kill_workers; echo "[parallel] SIGTERM"; exit 143' TERM

# ============================================================================
#  第一阶段：多卡并行推理
# ============================================================================
# 为每张 GPU 启动一个独立的推理进程（后台运行）
# model_vqa_loader 通过 --num-chunks 和 --chunk-idx 自动对问题文件做切分：
#   总共分为 NUM_GPUS 个 chunk，当前进程只处理第 i 个 chunk

for i in $(seq 0 $((NUM_GPUS - 1))); do
  echo "[parallel] GPU $i  chunk ${i}/${NUM_GPUS}"

  # CUDA_VISIBLE_DEVICES=$i  : 将第 i 张物理 GPU 映射为当前进程可见的唯一 GPU
  # DISABLE_PRUMERGE=1       : 确保子进程也禁用 PruMerge
  # &                        : 在后台运行，不阻塞循环
  CUDA_VISIBLE_DEVICES=$i DISABLE_PRUMERGE=1 python -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file "${PART_DIR}/part_${i}.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --num-chunks "${NUM_GPUS}" \
    --chunk-idx "${i}" &

  # 记录刚启动的后台进程 PID（$! 是最近一个后台进程的 PID）
  pids+=($!)
done

# ============================================================================
#  等待所有 worker 完成
# ============================================================================

wait_fail=0
# 临时关闭 set -e，因为 wait 在子进程失败时会返回非零，我们需要手动处理
set +e
for pid in "${pids[@]}"; do
  # 阻塞等待每个 worker 进程结束
  wait "$pid"
  s=$?  # 获取该 worker 的退出码
  if [[ $s -ne 0 ]]; then
    # 有 worker 失败：记录退出码并跳出循环
    wait_fail=$s
    echo "[parallel] worker pid=${pid} 退出码=${s}"
    break
  fi
done
# 恢复 set -e
set -e

# 如果有 worker 失败，杀掉其余还在运行的 worker 并退出
if [[ $wait_fail -ne 0 ]]; then
  _parallel_kill_workers
  exit "$wait_fail"
fi

# 所有 worker 都成功完成，移除信号捕获（恢复默认行为）
trap - INT TERM

# ============================================================================
#  第二阶段：合并各 chunk 结果
# ============================================================================

echo "[parallel] 合并到 ${MERGED_JSONL}"
# `: >` 清空目标文件（如果存在则截断为空，不存在则创建）
: > "${MERGED_JSONL}"
# 按 chunk 编号顺序依次追加，确保合并后的 jsonl 行序与原始问题文件一致
for i in $(seq 0 $((NUM_GPUS - 1))); do
  cat "${PART_DIR}/part_${i}.jsonl" >> "${MERGED_JSONL}"
done

# ============================================================================
#  第三阶段：计算评测指标
# ============================================================================
# 使用官方 TextVQA 评测脚本，将模型预测结果与标注答案对比，输出准确率

python -m llava.eval.eval_textvqa \
  --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
  --result-file "${MERGED_JSONL}"

echo "[parallel] 完成。"
