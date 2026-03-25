# coding=utf-8
"""双差分视觉 token 打分，供 FastV 风格剪枝使用。

相比 FastV（只用最后一个 token 行打分），本方法从两个无偏视角综合评估：
  c_i — 图像内部：token i 被其它图像 token 关注的程度（因果校正后）
  r_i — 问题→图像：token i 被所有问题 token 关注的程度（天然无偏）
"""
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
    双差分对"图像 token 段"的打分（因果校正版）。

    c_i — 图像内部重要性：
      在图像内注意力子矩阵 m 中，token j 被后续图像 token 关注的列和，
      除以因果可见数 (n_eff - j) 做归一化，消除"靠前=被更多人看到"的位置偏差，
      再做留一差分。

    r_i — 问题→图像重要性：
      只取图像块之后的 token（问题/文本）对图像列的注意力，
      这些 token 能看到所有图像 token → 天然无因果偏差。
      聚合后做留一差分。等效于 FastV "最后 token 行" 的多投票者版本。

    最终  s_i = c_i + lambda_tt * r_i
    """
    if attn_avg.dim() != 2 or attn_avg.shape[0] != attn_avg.shape[1]:
        raise ValueError(f"attn_avg must be square 2D, got {tuple(attn_avg.shape)}")
    L = attn_avg.shape[0]
    img_start = int(sys_len)
    img_end = min(img_start + int(n_img), L)
    n_eff = img_end - img_start
    if n_eff <= 0:
        return torch.zeros(0, device=attn_avg.device, dtype=attn_avg.dtype)

    device = attn_avg.device
    dtype = attn_avg.dtype

    # ---- c_i：图像内部重要性（对称化） ----
    # 由于 decoder 是因果的，图像内部子矩阵 m 天然是非对称的（下三角为主）。
    # 按你的要求：只对图像内部先做对称化 m_sym = (m + m^T) / 2，
    # 再统计列和得到重要性分布。
    m = attn_avg[img_start:img_end, img_start:img_end]                 # [n_eff, n_eff]
    m_sym = (m + m.transpose(0, 1)) / 2.0                            # [n_eff, n_eff]
    col_sums_img = m_sym.sum(dim=0)                                   # [n_eff]
    # _ate_diff_1d 对线性缩放不敏感（只做线性变换+居中），这里保持数值尺度稳定
    col_means_img = col_sums_img / float(n_eff)                        # [n_eff]
    c = _ate_diff_1d(col_means_img)                                   # [n_eff]

    # ---- r_i：问题/文本 → 图像重要性（天然无偏） ----
    # 只取图像块之后的 token（问题、文本）：它们能看到所有图像 token，无因果偏差
    n_q = L - img_end
    if n_q > 0:
        q_to_img = attn_avg[img_end:, img_start:img_end]         # [n_q, n_eff]
        q_col_sums = q_to_img.sum(dim=0)                         # [n_eff]
        r = _ate_diff_1d(q_col_sums)                              # [n_eff]
    else:
        r = torch.zeros(n_eff, device=device, dtype=dtype)

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
