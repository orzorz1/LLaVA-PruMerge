# coding=utf-8
"""双差分视觉 token 打分，供 FastV 风格剪枝使用。"""
from __future__ import annotations

import torch


def _ate_diff_1d(x: torch.Tensor) -> torch.Tensor:
    """留一差分：score_i = x_i - Avg_{k!=i}(x_k) = (n * x_i - sum(x)) / (n - 1)"""
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
    """
    双差分对"图像 token 段"的打分。

    公式（i, j, k 均为图像 token 的局部索引）：
      图像内差分  S_ij = A_ij - Avg_{k!=i}(A_kj)
                  c_i  = sum_j S_ij

      问题→图像差分  D_qi = A_qi - Avg_{k!=i}(A_qk)   (q 为任意 query token)
                     r_i  = sum_q D_qi

      最终得分  s_i = c_i + lambda_tt * r_i

    化简可知：
      c_i = _ate_diff_1d( m.sum(dim=1) )[i]        m 为图像内注意力子矩阵
      r_i = _ate_diff_1d( img_cols.sum(dim=0) )[i]  img_cols 为所有 token 到图像列的注意力
    """
    if attn_avg.dim() != 2 or attn_avg.shape[0] != attn_avg.shape[1]:
        raise ValueError(f"attn_avg must be square 2D, got {tuple(attn_avg.shape)}")
    l_max = attn_avg.shape[0]
    img_start = int(sys_len)
    img_end = min(img_start + int(n_img), l_max)
    n_eff = img_end - img_start
    if n_eff <= 0:
        return torch.zeros(0, device=attn_avg.device, dtype=attn_avg.dtype)

    # ---- c_i：图像内部差分 ----
    # m[i,j] = 图像 token i 对图像 token j 的注意力
    m = attn_avg[img_start:img_end, img_start:img_end]       # [n_eff, n_eff]
    # row_sums[i] = sum_j A_ij，第 i 个图像 token 对所有图像 token 的出注意力总和
    row_sums = m.sum(dim=1)                                   # [n_eff]
    c = _ate_diff_1d(row_sums)                                # [n_eff]

    # ---- r_i：所有 token → 图像的差分 ----
    # img_cols[q, i] = 第 q 个 token 对第 i 个图像 token 的注意力
    img_cols = attn_avg[:, img_start:img_end]                 # [L, n_eff]
    # col_sums[i] = sum_q A_qi，图像 token i 从所有 query 收到的注意力总和
    col_sums = img_cols.sum(dim=0)                            # [n_eff]
    r = _ate_diff_1d(col_sums)                                # [n_eff]

    return c + float(lambda_tt) * r


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
    """
    给出图像 token 段中"需要保留的 topk 全局索引"。

    use_diff=True  → 用 fastv_diff_image_scores 做双差分 scores
    use_diff=False → 退回原始 FastV：用 attn_avg[question, img] 作为 scores
    """
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
