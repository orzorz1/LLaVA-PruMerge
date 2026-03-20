#    Copyright 2023 Haotian Liu
#
#    Standalone builder for FastV-enabled LLaVA (see llava_llama_fastv.py).
#    Original builder.py is unchanged.

import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from llava.model import LlavaMPTForCausalLM
from llava.model.language_model.llava_llama_fastv import LlavaLlamaForCausalLMFastV
from llava.model.language_model.llama_model_fastv import configure_llama_fastv
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model_fastv(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    fastv_k=2,
    fastv_r=0.5,
    fastv_sys_len=35,
    fastv_image_len=576,
    fastv_rope_table_len=1000,
    fastv_rope_positions_after_prune="relative",
    **kwargs,
):
    """Same as ``load_pretrained_model`` but uses ``LlavaLlamaForCausalLMFastV`` and enables FastV on the LLaMA stack."""
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if "llava" in model_name.lower():
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument."
            )
        if "lora" in model_name.lower() and model_base is not None:
            model_path = os.path.abspath(os.path.expanduser(model_path))
            _local = os.path.isdir(model_path)
            if _local and os.path.isfile(os.path.join(model_path, "config.json")):
                lora_cfg_pretrained = AutoConfig.from_pretrained(model_path, local_files_only=True)
            else:
                lora_cfg_pretrained = AutoConfig.from_pretrained("liuhaotian/llava-v1.5-7b-lora")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("Loading LLaVA (FastV stack) from base model...")
            model = LlavaLlamaForCausalLMFastV.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                )

            print("Loading additional LLaVA weights...")
            non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
            if os.path.exists(non_lora_path):
                non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
            else:
                from huggingface_hub import hf_hub_download

                cache_file = hf_hub_download(
                    repo_id="liuhaotian/llava-v1.5-7b-lora", filename="non_lora_trainables.bin"
                )
                non_lora_trainables = torch.load(cache_file, map_location="cpu")
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            if _local:
                import huggingface_hub.utils._validators as _hub_validators

                _orig_validate = _hub_validators.validate_repo_id

                def _allow_local_path(x):
                    if os.path.isdir(os.path.expanduser(str(x))) or (
                        isinstance(x, str) and (x.startswith("/") or os.path.exists(x))
                    ):
                        return
                    _orig_validate(x)

                _hub_validators.validate_repo_id = _allow_local_path
                try:
                    model = PeftModel.from_pretrained(model, model_path, local_files_only=True)
                finally:
                    _hub_validators.validate_repo_id = _orig_validate
            else:
                model = PeftModel.from_pretrained(model, model_path, local_files_only=False)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        elif model_base is not None:
            print("Loading LLaVA (FastV stack) from base model...")
            if "mpt" in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(
                        os.path.join(model_base, "configuration_mpt.py"),
                        os.path.join(model_path, "configuration_mpt.py"),
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                _local = os.path.isdir(os.path.expanduser(model_path))
                cfg_pretrained = AutoConfig.from_pretrained(
                    model_path, trust_remote_code=True, local_files_only=_local
                )
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                _local = os.path.isdir(os.path.expanduser(model_path))
                cfg_pretrained = AutoConfig.from_pretrained(model_path, local_files_only=_local)
                model = LlavaLlamaForCausalLMFastV.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )

            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLMFastV.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        if model_base is not None:
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if "llava" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

        if "mpt" not in model_name.lower():
            configure_llama_fastv(
                model.model,
                fastv_k=fastv_k,
                fastv_r=fastv_r,
                sys_len=fastv_sys_len,
                image_len=fastv_image_len,
                inplace=True,
                rope_table_len=fastv_rope_table_len,
                rope_positions_after_prune=fastv_rope_positions_after_prune,
            )
            print(
                f"[FastV] enabled: k={fastv_k}, r={fastv_r}, sys_len={fastv_sys_len}, "
                f"image_len={fastv_image_len}, keep_tokens={max(1, int(round(fastv_image_len * (1.0 - fastv_r))))}, "
                f"rope_table_len={fastv_rope_table_len}, rope_pos_after_prune={fastv_rope_positions_after_prune}"
            )

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
