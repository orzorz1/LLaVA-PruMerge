#!/bin/bash

# TextVQA 评测：LLaMA 侧 FastV token 剪枝（见 llava/model/language_model/llama_model_fastv.py）
# 视觉塔：clip_encoder.py。本脚本默认 export DISABLE_PRUMERGE=1（576 patch，与论文 FastV 设定一致）。
# 若去掉该行则走 PruMerge 压缩路径；FastV 侧已按运行期真实 N 与安全 topk 处理，两者都可跑。
#
# 在项目根目录执行：
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_fastv/textvqa.sh

MODEL_PATH="liuhaotian/llava-v1.5-7b"
# MODEL_PATH="/path/to/local/llava-v1.5-7b"

# 可选：原始 CLIP 576 tokens（禁用 PruMerge）
export DISABLE_PRUMERGE=1

python -m llava.eval.model_vqa_loader_fastv \
    --model-path "${MODEL_PATH}" \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-fastv.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --fastv-k 2 \
    --fastv-r 0.5 \
    --fastv-sys-len 35 \
    --fastv-image-len 576

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-fastv.jsonl
