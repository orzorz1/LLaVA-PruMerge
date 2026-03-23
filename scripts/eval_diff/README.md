# FastV + diff 评测脚本

与 [`scripts/eval_fastv`](../eval_fastv/README.md) **平行**：独立 `builder` / `llava_llama` / `llama_model` / 评测入口，**不修改**原版 `builder.py`、`model_vqa_loader.py`。

## 与 eval_fastv 的差异

| 项目 | eval_fastv | eval_diff |
|------|-------------|-----------|
| 视觉 token 排名 | 最后一 token 对图像注意力 | 双差分（question-token + token-token ATE），见 `llava/model/language_model/fastv_diff.py` |
| Builder | `builder_fastv.load_pretrained_model_fastv` | `builder_diff.load_pretrained_model_diff` |
| 推理入口 | `llava.eval.model_vqa_loader_fastv` | `llava.eval.model_vqa_loader_diff` |
| LLaMA 骨干 | `LlamaModelFastV` | `LlamaModelFastVDiff` |

新增/对照文件：

| 路径 | 说明 |
|------|------|
| `llava/model/language_model/fastv_diff.py` | 差分打分与 `fastv_image_topk_indices` |
| `llava/model/language_model/llama_model_diff.py` | `LlamaModelFastVDiff` + `configure_llama_fastv_diff` |
| `llava/model/language_model/llava_llama_diff.py` | `LlavaLlamaForCausalLMFastVDiff` |
| `llava/model/builder_diff.py` | `load_pretrained_model_diff` |
| `llava/eval/model_vqa_loader_diff.py` | TextVQA 等评测 CLI |

## TextVQA 单机

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diff/textvqa.sh
```

- 结果默认：`playground/data/eval/textvqa/answers/llava-v1.5-7b-fastv-diff.jsonl`
- 环境变量 **`FASTV_DIFF_LAMBDA_TT`**：token-token 项权重（默认 1.0，与 `textvqa.sh` 内一致）

## TextVQA 多卡并行

```bash
NUM_GPUS=4 bash scripts/eval_diff/textvqa_parallel.sh
```

- **`MERGED_JSONL`**、**`PART_DIR`**、**`NUM_GPUS`**、**`MODEL_PATH`** 与 `eval_fastv` 用法相同
- **`FASTV_DIFF_LAMBDA_TT`**：各 worker 共用

## CLI 额外参数（`model_vqa_loader_diff`）

- **`--no-fastv-diff`**：关闭双差分，与 `eval_fastv` 的排名方式一致（仍走 `LlamaModelFastVDiff` 与同一套剪枝流程）
- **`--fastv-diff-lambda-tt`**：默认 `1.0`

其余 `--fastv-k` / `--fastv-r` / `--fastv-sys-len` / `--fastv-image-len` / RoPE 相关说明见 [`eval_fastv/README.md`](../eval_fastv/README.md)。

## 注意

- 同样需要 **`generate(..., use_cache=False)`**（脚本已设置）
- PruMerge 576 token：可 `export DISABLE_PRUMERGE=1`，与 eval_fastv 一致
