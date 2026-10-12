from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from models.affinity import ConfidenceAwareAffinity
from models.causal_residual import CausalResidualModule
from models.corruption import StructuredCorruptionGenerator
from models.graph_encoder import GraphEncoder
from models.temporal_encoder import TemporalEncoder
from utils.graph_utils import adjacency_to_tensor


class ContrastiveCausalModel(nn.Module):
    """Main model for window-level causal contrastive anomaly detection."""

    def __init__(
        self,
        num_nodes: int,
        window_size: int,
        hidden_dim: int,
        proj_dim: int,
        gru_layers: int,
        graph_layers: int,
        temporal_encoder_type: str,
        graph_encoder_type: str,
        predictor_hidden_dim: int,
        predictor_dropout: float,
        confidence_sigma: float,
        affinity_cfg: Dict[str, Any],
        corruption_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.confidence_sigma = max(confidence_sigma, 1e-6)

        self.temporal_encoder = TemporalEncoder(
            hidden_dim=hidden_dim,
            gru_layers=gru_layers,
            dropout=predictor_dropout,
            encoder_type=temporal_encoder_type,
            window_size=window_size,
        )
        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim,
            num_layers=graph_layers,
            encoder_type=graph_encoder_type,
        )
        self.causal_residual = CausalResidualModule(
            num_nodes=num_nodes,
            window_size=window_size,
            hidden_dim=predictor_hidden_dim,
            dropout=predictor_dropout,
        )
        # Retained for compatibility with optional legacy exports.
        self.affinity_builder = ConfidenceAwareAffinity(**affinity_cfg)
        self.corruption_generator = StructuredCorruptionGenerator(**corruption_cfg)
        self.node_projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.window_projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any], num_nodes: int) -> "ContrastiveCausalModel":
        """Build the complete model from a config dictionary."""

        model_cfg = config["model"]
        return cls(
            num_nodes=num_nodes,
            window_size=int(config["data"]["window_size"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            proj_dim=int(model_cfg["proj_dim"]),
            gru_layers=int(model_cfg["gru_layers"]),
            graph_layers=int(model_cfg["graph_layers"]),
            temporal_encoder_type=str(model_cfg.get("temporal_encoder", "gru")),
            graph_encoder_type=str(model_cfg.get("graph_encoder", "directed")),
            predictor_hidden_dim=int(model_cfg["predictor_hidden_dim"]),
            predictor_dropout=float(model_cfg.get("predictor_dropout", 0.1)),
            confidence_sigma=float(model_cfg["confidence_sigma"]),
            affinity_cfg=config["affinity"],
            corruption_cfg=config["corruption"],
        )

    def encode_windows(self, x: torch.Tensor, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode a batch of windows into node and window embeddings.

        Args:
            x: Input windows of shape ``[B, T, N]``.
            adjacency: Directed graph prior of shape ``[N, N]``.

        Returns:
            A dictionary containing:
            - ``node_embeddings``: ``[B, N, P]``
            - ``window_embeddings``: ``[B, P]``
            - intermediate node/window representations
        """

        temporal_out = self.temporal_encoder(x)
        graph_out = self.graph_encoder(temporal_out["node_repr"], adjacency)

        node_projection = self.node_projection_head(graph_out["node_repr"])
        node_embeddings = F.normalize(node_projection, dim=-1, eps=1e-8)

        window_projection = self.window_projection_head(graph_out["global_repr"])
        window_embeddings = F.normalize(window_projection, dim=-1, eps=1e-8)

        return {
            "temporal_node_repr": temporal_out["node_repr"],
            "temporal_global_repr": temporal_out["global_repr"],
            "graph_node_repr": graph_out["node_repr"],
            "graph_global_repr": graph_out["global_repr"],
            "node_embeddings": node_embeddings,
            "window_embeddings": window_embeddings,
        }

    def encode_window_corruptions(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """Generate window-level structured negatives and encode them."""

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)
        embedding_dim = self.window_projection_head[-1].out_features
        negatives_per_window = self.corruption_generator.num_window_negatives

        if x.shape[0] == 0:
            return {
                "corrupted_windows": x.new_empty((0, x.shape[1], x.shape[2])),
                "corrupted_modes": [],
                "target_node_indices": x.new_empty((0, negatives_per_window), dtype=torch.long),
                "root_node_indices": x.new_empty((0, negatives_per_window), dtype=torch.long),
                "subgraph_sizes": x.new_empty((0, negatives_per_window), dtype=torch.long),
                "corrupted_window_embeddings": x.new_empty((0, negatives_per_window, embedding_dim)),
            }

        (
            corrupted_windows,
            corrupted_modes,
            root_node_indices,
            subgraph_sizes,
        ) = self.corruption_generator.generate_window_corruptions(
            x=x,
            adjacency=adjacency_t,
            deterministic=deterministic,
        )
        corrupted_encoded = self.encode_windows(corrupted_windows, adjacency_t)
        corrupted_window_embeddings = corrupted_encoded["window_embeddings"].view(x.shape[0], negatives_per_window, -1)

        return {
            "corrupted_windows": corrupted_windows,
            "corrupted_modes": corrupted_modes,
            # Kept as an alias for backward compatibility with the old node-based naming.
            "target_node_indices": root_node_indices.view(x.shape[0], negatives_per_window),
            "root_node_indices": root_node_indices.view(x.shape[0], negatives_per_window),
            "subgraph_sizes": subgraph_sizes.view(x.shape[0], negatives_per_window),
            "corrupted_window_embeddings": corrupted_window_embeddings,
        }

    def encode_targeted_corruptions(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        target_sample_indices: torch.Tensor,
        target_node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """Generate one targeted negative per positive node and encode it.

        Args:
            x: Original batch of windows with shape ``[B, T, N]``.
            adjacency: Directed graph prior ``[N, N]``.
            target_sample_indices: Positive window indices ``[M]``.
            target_node_indices: Positive node indices ``[M]``.

        Returns:
            ``corrupted_node_embeddings`` is aligned with the flattened positive
            node list and therefore has shape ``[M, P]``.
        """

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)
        target_sample_indices = target_sample_indices.to(device=x.device, dtype=torch.long).view(-1)
        target_node_indices = target_node_indices.to(device=x.device, dtype=torch.long).view(-1)

        if target_sample_indices.numel() == 0:
            return {
                "corrupted_windows": x.new_empty((0, x.shape[1], x.shape[2])),
                "corrupted_modes": [],
                "corrupted_node_embeddings": x.new_empty((0, self.window_projection_head[-1].out_features)),
            }

        corrupted_windows, corrupted_modes = self.corruption_generator.generate_node_corruptions(
            x=x,
            adjacency=adjacency_t,
            sample_indices=target_sample_indices,
            node_indices=target_node_indices,
            deterministic=deterministic,
        )
        corrupted_encoded = self.encode_windows(corrupted_windows, adjacency_t)
        row_indices = torch.arange(corrupted_windows.shape[0], device=x.device)
        corrupted_node_embeddings = corrupted_encoded["node_embeddings"][row_indices, target_node_indices]

        return {
            "corrupted_windows": corrupted_windows,
            "corrupted_modes": corrupted_modes,
            "corrupted_node_embeddings": corrupted_node_embeddings,
        }

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        generate_corrupted: bool = False,
        corruption_deterministic: bool = False,
    ) -> Dict[str, Any]:
        """Run the shared encoder stack and the causal residual module.

        ``generate_corrupted`` is kept only for backward compatibility with
        existing call sites; node-targeted corruption generation is handled by
        ``encode_targeted_corruptions()`` after positive nodes are selected.
        """

        del generate_corrupted, corruption_deterministic

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)
        encoded = self.encode_windows(x, adjacency_t)
        causal_out = self.causal_residual(x, adjacency_t)

        causal_scores = causal_out["window_causal_score"]
        confidence_scores = torch.exp(-causal_scores / self.confidence_sigma)

        return {
            "node_embeddings": encoded["node_embeddings"],
            "window_embeddings": encoded["window_embeddings"],
            # Compatibility alias for legacy window-level code paths.
            "embedding": encoded["window_embeddings"],
            "temporal_node_repr": encoded["temporal_node_repr"],
            "graph_node_repr": encoded["graph_node_repr"],
            "global_repr": encoded["graph_global_repr"],
            "confidence_scores": confidence_scores,
            "causal_scores": causal_scores,
            "node_residuals": causal_out["node_residuals"],
            "causal_mse_loss": causal_out["mse_loss"],
        }
