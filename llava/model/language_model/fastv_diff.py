# coding=utf-8
"""双差分（ATE）视觉 token 打分，供 FastV 风格剪枝使用（与 FastV 仓库 ``fastv_diff.py`` 对齐）。"""
from __future__ import annotations

import torch


def _ate_diff_1d(x: torch.Tensor) -> torch.Tensor:
    # ATE (Adaptive Token Elimination) 的 1D 差分打分形式。
    # 给定一维向量 x（长度 n，对应一组 token 的某种注意力聚合分数），
    # 计算每个位置相对“全局均值”的差分量：
    #   score_i = (n * x_i - sum(x)) / (n - 1)
    # 直观上：它会把“比平均值高”的位置打高，把“低于平均”的位置打低，
    # 从而让 topk 更偏向“相对更显著/更有区分度”的 token。
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
    双差分（diff / ATE）对“图像 token 段”的打分。

    输入：
      - attn_avg: 注意力均值后的 2D 方阵，shape = [seq_len, seq_len]
                  attn_avg[i, j] 表示 token i 对 token j 的注意力强度（经过 head 平均）。
      - sys_len: 系统/文本前缀的 token 数，图像 token 段从索引 sys_len 开始。
      - n_img: 目标图像 token 数（会再 clamp 到真实序列上限）
      - lambda_tt: token-to-token(图像内部)差分项的权重
      - question_row_index: 选择注意力矩阵的哪一行作为 “question -> image” 的 question token
                            （FastV 风格一般传 -1，即最后一个 token 行）

    输出：
      - scores: 1D 向量，长度为实际可用的图像 token 数 n_eff；
                 每个元素是对应图像 token 的双差分得分 tau_q + lambda_tt * tau_tt。
    """
    if attn_avg.dim() != 2 or attn_avg.shape[0] != attn_avg.shape[1]:
        raise ValueError(f"attn_avg must be square 2D, got {tuple(attn_avg.shape)}")
    l_max = attn_avg.shape[0]
    img_start = int(sys_len)
    # img_end 是图像段的右边界（左闭右开），同时确保不超过 seq_len
    img_end = min(img_start + int(n_img), l_max)
    n_eff = img_end - img_start
    if n_eff <= 0:
        return torch.zeros(0, device=attn_avg.device, dtype=attn_avg.dtype)

    # 选择 question token 对应的注意力“行”
    # % l_max 的作用是：允许传入 -1 这种负索引时仍能落到合法区间。
    qi = question_row_index % l_max

    # question -> image：question token 到每个图像 token 的注意力分数（1D）
    q_to_img = attn_avg[qi, img_start:img_end]

    # image -> image：图像 token 段内部的注意力子矩阵
    m = attn_avg[img_start:img_end, img_start:img_end]

    # token-to-token(同伴)聚合：
    # peer[j] 表示图像 token j 在“同伴注意力”维度上的平均贡献。
    # 这里用 mean(dim=0) 对应“对每一列 token j 的平均（来自所有行 token i）”。
    peer = m.mean(dim=0)

    # tau_q：对 question->image 向量做 1D 差分
    tau_q = _ate_diff_1d(q_to_img)

    # tau_tt：对 token-to-token(同伴)向量做 1D 差分
    tau_tt = _ate_diff_1d(peer)

    # 最终双差分得分：同时考虑“与 question 的相关性（tau_q）”和
    # “在同伴竞争中的相对突出程度（tau_tt）”
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
    """
    给出图像 token 段中“需要保留的 topk 全局索引”。

    重要点：
      - 先在图像段内部计算 scores（长度 n_eff）
      - scores.topk 得到的是局部索引（范围 [0, n_eff)）
      - 再 + img_start 得到全局索引（范围 [sys_len, seq_len)）

    如果 use_diff=False，则退回原始 FastV：直接用 attn_avg[question, img] 作为 scores。
    如果 use_diff=True，则用 fastv_diff_image_scores 做双差分 scores。
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
        # 双差分（ATE）得分
        scores = fastv_diff_image_scores(
            attn_avg,
            sys_len,
            n_img,
            lambda_tt=lambda_tt,
            question_row_index=question_row_index,
        )
    else:
        # 原始 FastV 排名：question -> image 注意力直接 topk
        qi = question_row_index % l_max
        scores = attn_avg[qi, img_start:img_end]

    top_local = scores.topk(k, dim=-1).indices
    # top_local 是局部索引，需要映射回全局序列下标
    return top_local + img_start
