from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from utils.graph_utils import adjacency_to_tensor, compute_graph_role_features


class ConfidenceAwareAffinity(nn.Module):
    """构造窗口级的因果感知软正样本权重矩阵。"""

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.5,
        gamma: float = 0.3,
        topk: int | None = None,
        positive_ratio: float = 0.8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        del topk

        if not 0.0 < positive_ratio <= 1.0:
            raise ValueError("positive_ratio must satisfy 0 < positive_ratio <= 1.")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.positive_ratio = positive_ratio
        self.eps = eps

    @staticmethod
    def _pairwise_cosine_similarity(features: torch.Tensor) -> torch.Tensor:
        """计算两两余弦相似度，并映射到 ``[0, 1]``。"""

        normalized = torch.nn.functional.normalize(features, dim=-1, eps=1e-8)
        cosine = normalized @ normalized.transpose(0, 1)
        cosine = cosine.clamp(min=-1.0, max=1.0)
        return 0.5 * (cosine + 1.0)

    def _build_graph_similarity(self, node_residuals: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """根据残差签名构造图角色相似度。"""

        role_features = compute_graph_role_features(adjacency).to(node_residuals.device)
        graph_signatures = (node_residuals.unsqueeze(-1) * role_features.unsqueeze(0)).sum(dim=1)
        return self._pairwise_cosine_similarity(graph_signatures)

    def _build_ratio_mask(self, weighted_similarity: torch.Tensor) -> torch.Tensor:
        """按比例保留每个锚点窗口的候选正样本。"""

        batch_size = weighted_similarity.shape[0]
        mask = torch.zeros_like(weighted_similarity)
        if batch_size <= 1:
            return mask

        keep_count = int(math.ceil(self.positive_ratio * float(batch_size - 1)))
        keep_count = max(1, min(batch_size - 1, keep_count))

        scores = weighted_similarity.clone()
        eye = torch.eye(batch_size, device=weighted_similarity.device, dtype=torch.bool)
        scores = scores.masked_fill(eye, float("-inf"))
        keep_indices = torch.topk(scores, k=keep_count, dim=1).indices
        mask.scatter_(1, keep_indices, 1.0)
        mask.fill_diagonal_(0.0)
        return mask

    def forward(
        self,
        embeddings: torch.Tensor,
        node_residuals: torch.Tensor,
        confidence_scores: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """构造窗口级相似度、比例筛选掩码和软权重矩阵。"""

        adjacency_t = adjacency_to_tensor(adjacency, device=embeddings.device)

        s_state = self._pairwise_cosine_similarity(embeddings)
        s_mech = self._pairwise_cosine_similarity(node_residuals)
        s_graph = self._build_graph_similarity(node_residuals, adjacency_t)

        weight_sum = max(self.alpha + self.beta + self.gamma, self.eps)
        similarity = (
            self.alpha * s_graph
            + self.beta * s_state
            + self.gamma * s_mech
        ) / weight_sum
        similarity = similarity.clamp(min=0.0, max=1.0)

        confidence_matrix = confidence_scores.unsqueeze(1) * confidence_scores.unsqueeze(0)
        weighted_similarity = confidence_matrix * similarity

        mask = self._build_ratio_mask(weighted_similarity)
        affinity = mask * weighted_similarity
        row_sums = affinity.sum(dim=1, keepdim=True)
        affinity_norm = affinity / (row_sums + self.eps)
        positive_counts = mask.sum(dim=1)
        valid_window_mask = positive_counts > 0

        return {
            "S_graph": s_graph,
            "S_state": s_state,
            "S_mech": s_mech,
            "S": similarity,
            "C": confidence_matrix,
            "Q": weighted_similarity,
            "M": mask,
            "W": affinity,
            "W_norm": affinity_norm,
            "positive_counts": positive_counts,
            "valid_window_mask": valid_window_mask,
        }
