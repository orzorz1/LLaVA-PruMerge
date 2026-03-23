# Copyright 2023 Haotian Liu and FastV authors (PKU NLP).
# FastV logic adapted from https://github.com/pkunlp-icler/FastV (ECCV 2024).
# This file is a standalone extension for LLaVA-PruMerge; does not modify upstream transformers.

"""LlamaModel + FastV inplace visual-token pruning (transformers==4.31 compatible).

Requires ``output_attentions=True`` internally during prefill to read layer (K-1) attentions.
Use ``model.generate(..., use_cache=False)`` — KV cache is incompatible with inplace pruning.
"""

from typing import List, Optional, Tuple, Union

import torch

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel, logger


def patch_llama_rotary_emb_for_fastv(llama_inner: LlamaModel, rope_table_len: int = 1000) -> None:
    """对齐 FastV 官方思路 + 处理长 prompt。

    剪枝后 ``position_ids = keep_indexs``，索引可达**剪枝前**全长（TextVQA 带长 OCR 时可能 >1000）。
    仅 ``max(seq_len, rope_table_len)`` 仍会在 ``apply_rotary_pos_emb`` 里 `cos[position_ids]` 越界。
    因此在每条前向里用 ``_fastv_rope_index_ub = keep_indexs.max() + 1`` 动态抬高 RoPE 表长：
    ``seq_len = max(传入, rope_table_len, _fastv_rope_index_ub)``。
    """
    if getattr(llama_inner, "_fastv_rope_forward_patched", False):
        return
    rope_table_len = int(rope_table_len)
    for layer in llama_inner.layers:
        rope = layer.self_attn.rotary_emb
        if getattr(rope, "_fastv_rope_forward_patched", False):
            continue
        orig_forward = rope.forward

        def _wrapped_fwd(
            *args,
            __orig=orig_forward,
            __min_static=rope_table_len,
            __lm=llama_inner,
            **kwargs,
        ):
            need = __min_static
            dyn = getattr(__lm, "_fastv_rope_index_ub", None)
            if dyn is not None:
                need = max(need, int(dyn))
            if kwargs.get("seq_len", None) is not None:
                kwargs = dict(kwargs)
                kwargs["seq_len"] = max(int(kwargs["seq_len"]), need)
            return __orig(*args, **kwargs)

        rope.forward = _wrapped_fwd
        rope._fastv_rope_forward_patched = True
    llama_inner._fastv_rope_forward_patched = True
    llama_inner.fast_v_rope_table_len = rope_table_len


class LlamaModelFastV(LlamaModel):
    """Drop-in replacement for ``LlamaModel`` with optional FastV (inplace only)."""

    def __init__(self, config):
        super().__init__(config)
        # Populated by ``configure_llama_fastv`` after load; all off by default.
        self.use_fast_v = False
        self.fast_v_inplace = True
        self.fast_v_sys_length = 35
        self.fast_v_image_token_length = 576
        self.fast_v_attention_rank = 288
        self.fast_v_agg_layer = 2
        # "relative"（默认）：剪枝后 position_ids=0..L'-1，RoPE 稳定； "absolute"：与 FastV 仓库一致用 keep_indexs，需大张长 RoPE 表
        self.fast_v_rope_positions_after_prune = "relative"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        user_output_attentions = output_attentions

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_fastv = getattr(self, "use_fast_v", False) and getattr(self, "fast_v_inplace", True)

        if not use_fastv or past_key_values is not None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if use_cache:
            msg = (
                "FastV inplace pruning is not compatible with use_cache=True. "
                "For correct results use model.generate(..., use_cache=False)."
            )
            if hasattr(logger, "warning_once"):
                logger.warning_once(msg)
            else:
                logger.warning(msg)

        if self.gradient_checkpointing and self.training:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=user_output_attentions if user_output_attentions is not None else output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Need attention weights from layer (K-1) to select tokens before layer K.
        output_attentions = True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                _w = "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                if hasattr(logger, "warning_once"):
                    logger.warning_once(_w)
                else:
                    logger.warning(_w)
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        SYS_LENGTH = self.fast_v_sys_length
        IMAGE_TOKEN_LENGTH = self.fast_v_image_token_length
        ATTENTION_RANK = self.fast_v_attention_rank
        # AGG_LAYER = 配置里的 fastv_k（例如 2）：表示「从第几层开始做删减图像 token」。
        # 实现上：先用第 AGG_LAYER-1 层算出的注意力来打分，再在第 AGG_LAYER 层开头删掉不重要的图像 token。
        AGG_LAYER = self.fast_v_agg_layer
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        layer_outputs = None
        # 每条序列开始前清掉；若剪枝后需要扩大 RoPE 表长度，会在下面写入最大位置
        self._fastv_rope_index_ub = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                # 删减图像 token 需要「上一层」的注意力分数，所以：
                # 跑到「倒数第二层」（idx = AGG_LAYER-1）时，必须把这层的注意力算出来并保存；
                # 下一层（idx = AGG_LAYER）一进来，就用这份注意力给图像块打分、删掉一部分 token。
                need_attn_for_prune = idx == AGG_LAYER - 1
                layer_out_att = True if need_attn_for_prune else output_attentions

                if idx < AGG_LAYER:
                    new_attention_mask = attention_mask
                elif idx == AGG_LAYER:
                    if layer_outputs is None:
                        raise RuntimeError("FastV: expected previous layer outputs before agg layer")
                    last_layer_attention = layer_outputs[1]
                    # 多头先平均成一头，再取第 0 条样本（一般一次只跑一张图）→ 得到一张「谁在看谁」的方阵
                    last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]

                    # FastV：看「最后那个 token」（通常是问题末尾）对序列里每个位置的注意力，越大越重要
                    last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]

                    # 图像占多少个 token：默认 576；若开了 PruMerge 会少很多，用 encode_images 里记好的数量
                    n_img_cfg = IMAGE_TOKEN_LENGTH
                    n_img_rt = getattr(self, "_fastv_runtime_n_image", None)
                    if n_img_rt is not None:
                        n_img = int(n_img_rt)
                    else:
                        n_img = int(n_img_cfg)

                    # 别让图像段超出当前句子长度；至少留 1 个图像 token，避免后面算崩
                    max_img_span = max(0, seq_length_with_past - SYS_LENGTH)
                    n_img = max(1, min(n_img, max_img_span))

                    # 图像块在序列里到哪儿结束（从 SYS_LENGTH 开始数 n_img 个）
                    img_end = SYS_LENGTH + n_img

                    # 只截「问题末尾 → 图像这一块」的注意力，用来比大小
                    last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:img_end]

                    # 打算留几个图像 token：由 ATTENTION_RANK 定，但不能超过实际图像长度，至少 1 个
                    k_keep = min(ATTENTION_RANK, last_layer_attention_avg_last_tok_image.numel())
                    k_keep = max(k_keep, 1)

                    # 在图像块里挑注意力最高的 k_keep 个；下标要加回 SYS_LENGTH 才是整句里的位置
                    top_attention_rank_index = (
                        last_layer_attention_avg_last_tok_image.topk(k_keep, dim=-1).indices + SYS_LENGTH
                    )

                    # 拼起来：前面系统词 + 挑中的图像 + 图像后面的话（问题里图像后面的部分）→ 排序后按顺序裁剪
                    keep_indexs = torch.cat(
                        (
                            torch.arange(SYS_LENGTH, device=device),
                            top_attention_rank_index,
                            torch.arange(img_end, seq_length_with_past, device=device),
                        )
                    )
                    keep_indexs = keep_indexs.sort().values
                    new_seq_length = keep_indexs.shape[0]

                    # 只保留这些位置上的向量，相当于把序列变短了
                    hidden_states = hidden_states[:, keep_indexs, :]

                    # 裁剪后每个 token 的位置编号怎么写：
                    # relative（默认）：从 0 重新数，不容易越界；absolute：还用原句子里的位置号，需要 RoPE 表够长
                    pos_mode = getattr(self, "fast_v_rope_positions_after_prune", "relative")
                    if pos_mode == "absolute":
                        position_ids = keep_indexs.unsqueeze(0).expand(batch_size, -1)
                        self._fastv_rope_index_ub = int(keep_indexs.max().item()) + 1
                    else:
                        # 新序列从 0 开始编号：0,1,2,… 到 new_seq_length-1
                        position_ids = (
                            torch.arange(new_seq_length, device=device, dtype=torch.long)
                            .unsqueeze(0)
                            .expand(batch_size, -1)
                        )
                        self._fastv_rope_index_ub = None
                    new_attention_mask = self._prepare_decoder_attention_mask(
                        None, (batch_size, new_seq_length), inputs_embeds, 0
                    )
                else:
                    new_attention_mask = self._prepare_decoder_attention_mask(
                        None, (batch_size, hidden_states.shape[1]), inputs_embeds, 0
                    )

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=new_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=layer_out_att,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if layer_out_att:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not user_output_attentions:
            all_self_attns = None

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def configure_llama_fastv(
    llama_inner,
    *,
    fastv_k: int = 2,
    fastv_r: float = 0.5,
    sys_len: int = 35,
    image_len: int = 576,
    inplace: bool = True,
    rope_table_len: int = 1000,
    rope_positions_after_prune: str = "relative",
) -> None:
    """Attach FastV hyper-parameters to the inner LLaMA (``LlamaModelFastV`` instance)."""
    if fastv_k <= 0:
        raise ValueError("fastv_k (agg layer) must be > 0")
    if rope_positions_after_prune not in ("relative", "absolute"):
        raise ValueError("rope_positions_after_prune must be 'relative' or 'absolute'")
    llama_inner.use_fast_v = True
    llama_inner.fast_v_inplace = inplace
    llama_inner.fast_v_agg_layer = fastv_k
    llama_inner.fast_v_sys_length = sys_len
    llama_inner.fast_v_image_token_length = image_len
    llama_inner.fast_v_attention_rank = max(1, int(round(image_len * (1.0 - fastv_r))))
    llama_inner.fast_v_rope_positions_after_prune = rope_positions_after_prune
    patch_llama_rotary_emb_for_fastv(llama_inner, rope_table_len=rope_table_len)
