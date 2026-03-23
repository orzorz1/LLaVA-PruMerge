# coding=utf-8
"""双差分（ATE）视觉 token 打分，供 FastV 风格剪枝使用（与 FastV 仓库 ``fastv_diff.py`` 对齐）。"""
from __future__ import annotations

import torch


def _ate_diff_1d(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    if n <= 1:
        return torch.zeros_like(x)
    total = x.sum()
    return (n * x - total) / (n - 1)


def fastv_diff_image_scores(
    attn_avg: torch.Tensor,
    sys_len: int,
    n_img: int,
    *,
    lambda_tt: float = 1.0,
    question_row_index: int = -1,
) -> torch.Tensor:
    if attn_avg.dim() != 2 or attn_avg.shape[0] != attn_avg.shape[1]:
        raise ValueError(f"attn_avg must be square 2D, got {tuple(attn_avg.shape)}")
    l_max = attn_avg.shape[0]
    img_start = int(sys_len)
    img_end = min(img_start + int(n_img), l_max)
    n_eff = img_end - img_start
    if n_eff <= 0:
        return torch.zeros(0, device=attn_avg.device, dtype=attn_avg.dtype)

    qi = question_row_index % l_max
    q_to_img = attn_avg[qi, img_start:img_end]
    m = attn_avg[img_start:img_end, img_start:img_end]
    peer = m.mean(dim=0)
    tau_q = _ate_diff_1d(q_to_img)
    tau_tt = _ate_diff_1d(peer)
    return tau_q + float(lambda_tt) * tau_tt


def fastv_image_topk_indices(
    attn_avg: torch.Tensor,
    sys_len: int,
    n_img: int,
    attention_rank: int,
    *,
    use_diff: bool,
    lambda_tt: float = 1.0,
    question_row_index: int = -1,
) -> torch.Tensor:
    device = attn_avg.device
    l_max = attn_avg.shape[0]
    img_start = int(sys_len)
    img_end = min(img_start + int(n_img), l_max)
    n_eff = max(img_end - img_start, 0)
    k = min(int(attention_rank), n_eff) if n_eff else 0
    k = max(k, 1) if n_eff > 0 else 0

    if n_eff == 0:
        return torch.tensor([], device=device, dtype=torch.long)

    if use_diff:
        scores = fastv_diff_image_scores(
            attn_avg,
            sys_len,
            n_img,
            lambda_tt=lambda_tt,
            question_row_index=question_row_index,
        )
    else:
        qi = question_row_index % l_max
        scores = attn_avg[qi, img_start:img_end]

    top_local = scores.topk(k, dim=-1).indices
    return top_local + img_start
