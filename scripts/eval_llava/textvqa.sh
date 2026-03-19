#!/bin/bash

# 只使用原始 LLaVA-v1.5-7B 做 TextVQA 评测（不加载 PruMerge LoRA）
# 在项目根目录下运行：
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_llava/textvqa.sh

# 1）模型路径：可以是 HuggingFace 名字，也可以改成你本地下载后的目录
MODEL_PATH="liuhaotian/llava-v1.5-7b"

# 如果你在本地下好了模型（例如 /data/models/llava-v1.5-7b），可以改成：
# MODEL_PATH="/data/models/llava-v1.5-7b"

# 禁用 PruMerge token 压缩，使用原始 CLIP 视觉编码（576 tokens）
export DISABLE_PRUMERGE=1

DISABLE_PRUMERGE=1 python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-original.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-original.jsonl

