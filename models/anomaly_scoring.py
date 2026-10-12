from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from training.memory_bank import MemoryBank


class AnomalyScorer:
    """Window-level anomaly scoring for the window-centric detection pipeline."""

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        knn_k: int = 5,
        residual_eps: float = 1e-6,
    ) -> None:
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.knn_k = knn_k
        self.residual_eps = residual_eps

    def compute_window_embedding_score(
        self,
        window_embeddings: torch.Tensor,
        memory_bank: Optional["MemoryBank"],
    ) -> torch.Tensor:
        """Compute window-level kNN distance to the memory bank."""

        if window_embeddings.ndim != 2:
            raise ValueError("window_embeddings must have shape [B, P].")

        batch_size = window_embeddings.shape[0]
        scores = torch.zeros(batch_size, device=window_embeddings.device, dtype=window_embeddings.dtype)
        if memory_bank is None or memory_bank.is_empty():
            return scores

        bank_embeddings = memory_bank.get_embeddings(device=window_embeddings.device, dtype=window_embeddings.dtype)
        distances = torch.cdist(window_embeddings, bank_embeddings, p=2)
        k = min(self.knn_k, bank_embeddings.shape[0])
        knn_distances = torch.topk(distances, k=k, largest=False, dim=1).values
        return knn_distances.mean(dim=1)

    def standardize_causal_scores(
        self,
        causal_scores: torch.Tensor,
        causal_stats: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Standardize window-level causal scores with train-set mean/std."""

        if causal_stats is None:
            return causal_scores

        mean = causal_stats.get("mean")
        std = causal_stats.get("std")
        if mean is None or std is None:
            return causal_scores

        mean_t = torch.as_tensor(mean, device=causal_scores.device, dtype=causal_scores.dtype)
        std_t = torch.as_tensor(std, device=causal_scores.device, dtype=causal_scores.dtype)
        return (causal_scores - mean_t) / (std_t + self.residual_eps)

    def score(
        self,
        causal_scores: torch.Tensor,
        window_embeddings: torch.Tensor,
        memory_bank: Optional["MemoryBank"],
        causal_stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute window-level anomaly scores."""

        causal_zscores = self.standardize_causal_scores(causal_scores, causal_stats)
        embed_scores = self.compute_window_embedding_score(window_embeddings, memory_bank)
        window_scores = self.lambda1 * causal_zscores + self.lambda2 * embed_scores

        return {
            "window_scores": window_scores,
            "anomaly_score": window_scores,
            "A_embed": embed_scores,
            "causal_scores": causal_scores,
            "causal_zscores": causal_zscores,
        }
