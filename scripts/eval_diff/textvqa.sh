#!/bin/bash

# TextVQA：FastV inplace + 双差分视觉 token 打分（见 llava/model/language_model/llama_model_diff.py）
# 与 eval_fastv 平行；不改动 builder.py / model_vqa_loader.py。
#
# 在项目根目录执行：
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diff/textvqa.sh

MODEL_PATH="liuhaotian/llava-v1.5-7b"

export DISABLE_PRUMERGE=1

python -m llava.eval.model_vqa_loader_diff \
    --model-path "${MODEL_PATH}" \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-fastv-diff.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --fastv-k 2 \
    --fastv-r 0.5 \
    --fastv-sys-len 35 \
    --fastv-image-len 576 \
    --fastv-diff-lambda-tt "${FASTV_DIFF_LAMBDA_TT:-1.0}"

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-fastv-diff.jsonl
