# FastV 评测脚本

本目录脚本使用 **独立副本** 的模型与评测入口，不修改原有 `builder.py` / `llava_llama.py` / `model_vqa_loader.py`。

## 新增文件

| 路径 | 说明 |
|------|------|
| `llama_model_fastv.py` | 剪枝后默认 **`rope_positions_after_prune=relative`**：`position_ids = 0..L'-1`，避免 `cos[position_ids]` 越界；可选 **`absolute`** 对齐官方 `keep_indexs` + RoPE 补丁 |
| `llava/model/language_model/llava_llama_fastv.py` | `LlavaLlamaForCausalLMFastV`：使用上述 backbone，**未**注册到 `AutoModel`，避免覆盖原版 |
| `llava/model/builder_fastv.py` | `load_pretrained_model_fastv`：加载逻辑同 `builder.py`，但实例化 FastV 版并调用 `configure_llama_fastv` |
| `llava/eval/model_vqa_loader_fastv.py` | 评测入口；`generate(..., use_cache=False)`（FastV inplace 与 KV cache 不兼容） |

## TextVQA

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_fastv/textvqa.sh
```

### 多卡并行推理（最简单）

**「一进程一卡 + 按数据切块」**：评测入口已支持 `--num-chunks N --chunk-idx i`，无需改 Python。

```bash
# 4 张卡并行（自动用 GPU 0,1,2,3，各跑 1/4 数据，合并后再算 TextVQA accuracy）
NUM_GPUS=4 bash scripts/eval_fastv/textvqa_parallel.sh
```

- 环境变量 **`NUM_GPUS`**：并行进程数 = 使用的 GPU 数（默认 4）。
- 合并结果默认写到：`playground/data/eval/textvqa/answers/llava-v1.5-7b-fastv.jsonl`（可用 **`MERGED_JSONL`** 覆盖）。
- 分片临时文件目录：**`PART_DIR`**（默认 `answers/_fastv_parallel_parts/`）。
- 需要 **`--fastv-rope-table-len`** 等与单机脚本一致时，可自行编辑 `textvqa_parallel.sh` 里 `python -m` 那一行，与 `textvqa.sh` 对齐。

每张卡都会**单独加载一份模型**（显存 × 进程数）；若只想「模型拆到多卡、单进程」，用默认 `device_map="auto"` 即可，不要用本并行脚本。

对本脚本按 **Ctrl+C**（或 `kill` 脚本 PID）会 **SIGTERM 所有已启动的 python worker**，避免只停前台、其余进程仍占 GPU。若你是在**多个终端里各跑一条命令**，Ctrl+C 只会停当前终端那一个进程，其它终端需分别停或统一用本脚本起子进程。

### CUBLAS_STATUS_EXECUTION_FAILED / `cublasGemmEx`

多见于：**多进程 DataLoader + CUDA**、**显存吃紧/碎片化**、**device_map 多卡但输入仍在默认 cuda:0**。评测脚本已默认 **`--num-workers 0`**，并把 `input_ids` / `images` 放到 **`embed_tokens` 所在设备**、dtype 与 **`model.dtype`** 一致。若仍报错可试：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；确认并行任务 **未把多进程绑同一张卡**；单卡先试 `CUDA_VISIBLE_DEVICES=0`。

### `--fastv-sys-len` 还有用吗？怎么核对？

**有用。** FastV 把「图像 token 段」当成从第 `sys_len` 个位置开始；必须与真实多模态 prompt 里 **第一个图像占位符在 `input_ids` 中的下标**一致，否则剪枝会对着错误的注意力段。

- **多数情况不用改：** LLaVA v1.5 + TextVQA + `vicuna_v1` 一般为 **35**（`textvqa.sh` 里默认如此）。
- **换了对话模板 / conv-mode / 是否在问题前加 `<im_start>` 等** 时，请按下式自检（与 `model_vqa_loader_fastv.py` 里拼 `prompt` 的方式保持一致）：

```python
import json
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.eval.model_vqa_loader_fastv import build_textvqa_prompt

# 已加载 tokenizer 与 model（model.config 含 mm_use_im_start_end）
line = json.loads(open("./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl").readline())
prompt = build_textvqa_prompt(line, model.config, "vicuna_v1")  # conv-mode 与评测一致
ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").view(-1)
fastv_sys_len = int((ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0].item())
print("--fastv-sys-len 应设为:", fastv_sys_len)
```

说明：`tokenizer_image_token` 得到的是 **1D** `input_ids`，必须对 **整段** `ids` 找 `IMAGE_TOKEN_INDEX`（-200），不能写 `input_ids[0]`（那只是第一个 token）。

## 超参数

- `--fastv-k`：在第 `k` 层前完成剪枝（与官方 FastV 中 K 含义一致；实现上使用第 `k-1` 层注意力，在第 `k` 层前裁剪）。
- `--fastv-r`：丢弃的视觉 token **比例**（保留 `(1-r)×576` 个）。
- `--fastv-sys-len`：图像 token 区段前的文本 token 数，需与 `conv-mode` 下真实 prompt **逐 token 对齐**。若设得**大于**真实前缀长度，则「以为的图像起点」偏右，可用序列 `seq_len - sys_len` 会 **短于 576**，旧实现仍按 `topk(288)` 裁剪，会出现 **`k` 大于切片长度 → CUDA device-side assert**（与是否开启 PruMerge **无关**）。
- `--fastv-image-len`：图像 token 数（默认 576）。
- `--fastv-rope-table-len`：RoPE `seq_len` **下限**（默认 1000）。在 **`--fastv-rope-positions-after-prune absolute`** 时与官方大张表思路一致，长 prompt 可调大。
- `--fastv-rope-positions-after-prune`：默认 **`relative`**（推荐，`generate`/长序列更稳）；设 **`absolute`** 则剪枝后用原序列下标（对齐 FastV 参考实现，依赖 RoPE 补丁与较长表）。

## 与 PruMerge 的关系

视觉塔仍使用 `clip_encoder.py`。若需「576 全 token + 仅 FastV」，在 shell 中设置 `export DISABLE_PRUMERGE=1`（见 `scripts/eval_llava/textvqa.sh` 说明）。

剪枝实现会：**(1)** 用 `encode_images` 得到的真实视觉 token 数 \(N\)（关 PruMerge 时通常仍为 576）；**(2)** 用 `min(N, seq_len - sys_len)` 约束可用图像段；**(3)** `topk` 的 \(k=\min(\text{rank}, \text{切片长度})\)。因此 **PruMerge 变小 N** 与 **`--fastv-sys-len` 偏大** 两类情况都不会再因 `k` 越界崩掉。`--fastv-image-len` 仍作回退。
