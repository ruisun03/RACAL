from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from utils.graph_utils import adjacency_to_tensor, normalize_directed_adjacency


class DirectedGraphLayer(nn.Module):
    """单层有向消息传递层。

    每个节点的更新同时融合三路信息：
    - 节点自身当前表示
    - 来自父节点方向的消息
    - 来自子节点方向的反向上下文

    这样做能在保持结构轻量的同时，把边方向真实纳入更新规则。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.in_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, node_repr: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """执行一步有向图消息传递。

        参数：
            node_repr: 形状为 ``[B, N, D]`` 的节点表示。
            adjacency: 形状为 ``[N, N]`` 的有向图，采用
                ``adjacency[src, dst] = 1`` 这套方向约定。
        """

        a_in, a_out = normalize_directed_adjacency(adjacency, add_self_loops=True)

        # ``inbound_messages[:, j]`` 聚合所有指向节点 j 的父节点消息。
        inbound_messages = torch.einsum("ij,bid->bjd", a_in, node_repr)

        # ``outbound_messages[:, j]`` 聚合节点 j 的子节点带来的反向上下文，
        # 让下游效应也能反馈到当前节点表示。
        outbound_messages = torch.einsum("ji,bid->bjd", a_out, node_repr)

        # 对自身/父节点/子节点三路信息分别做线性变换，再通过残差连接和层归一化
        # 提升训练稳定性。
        updated = (
            self.self_linear(node_repr)
            + self.in_linear(inbound_messages)
            + self.out_linear(outbound_messages)
        )
        updated = self.activation(updated)
        return self.norm(node_repr + updated)


class GCNGraphLayer(nn.Module):
    """Dense GCN layer using an undirected normalized graph."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, node_repr: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.float()
        num_nodes = adjacency.shape[0]
        graph = ((adjacency + adjacency.T) > 0).to(dtype=node_repr.dtype, device=node_repr.device)
        graph = graph + torch.eye(num_nodes, device=node_repr.device, dtype=node_repr.dtype)
        degree = graph.sum(dim=1).clamp(min=1.0)
        degree_inv_sqrt = torch.rsqrt(degree)
        normalized = degree_inv_sqrt.unsqueeze(1) * graph * degree_inv_sqrt.unsqueeze(0)

        aggregated = torch.einsum("ij,bjd->bid", normalized, node_repr)
        updated = self.activation(self.linear(aggregated))
        return self.norm(node_repr + updated)


class GATGraphLayer(nn.Module):
    """Single-head dense graph attention layer over directed parent messages."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.node_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target_attention = nn.Linear(hidden_dim, 1, bias=False)
        self.source_attention = nn.Linear(hidden_dim, 1, bias=False)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU(0.2)
        self.output_activation = nn.GELU()

    def forward(self, node_repr: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.float()
        num_nodes = adjacency.shape[0]
        # Rows are targets and columns are source nodes. ``adjacency[src, dst]``
        # therefore becomes ``message_mask[dst, src]`` for attention pooling.
        message_mask = adjacency.T > 0
        message_mask = message_mask | torch.eye(num_nodes, device=adjacency.device, dtype=torch.bool)
        message_mask = message_mask.to(device=node_repr.device)

        projected = self.node_linear(node_repr)
        target_logits = self.target_attention(projected)
        source_logits = self.source_attention(projected).transpose(1, 2)
        logits = self.activation(target_logits + source_logits)
        logits = logits.masked_fill(~message_mask.unsqueeze(0), torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)

        aggregated = torch.matmul(attention, projected)
        updated = self.output_activation(self.output_linear(aggregated))
        return self.norm(node_repr + updated)


class GraphEncoder(nn.Module):
    """利用固定图先验细化时序节点表示。"""

    def __init__(self, hidden_dim: int, num_layers: int = 2, encoder_type: str = "directed") -> None:
        super().__init__()
        encoder_type = encoder_type.lower()
        self.encoder_type = encoder_type
        if encoder_type in {"directed", "directed_gnn"}:
            layer_cls = DirectedGraphLayer
        elif encoder_type == "gcn":
            layer_cls = GCNGraphLayer
        elif encoder_type == "gat":
            layer_cls = GATGraphLayer
        else:
            raise ValueError(
                f"Unsupported graph encoder: {encoder_type}. "
                "Expected one of ['directed', 'gcn', 'gat']."
            )

        self.layers = nn.ModuleList([layer_cls(hidden_dim) for _ in range(num_layers)])
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_repr: torch.Tensor, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        """堆叠若干有向图层，并返回节点级和窗口级表示。"""

        adjacency_t = adjacency_to_tensor(adjacency, device=node_repr.device)
        hidden = node_repr

        # 多层消息传递会让每个节点逐步吸收更丰富的图上下文，同时通过残差连接
        # 保留初始时序编码结果。
        for layer in self.layers:
            hidden = layer(hidden, adjacency_t)

        # 对节点表示做均值池化，得到窗口级图表示，后续再送入投影头构造
        # 对比学习用的表示向量。
        global_repr = self.global_projection(hidden.mean(dim=1))
        return {
            "node_repr": hidden,
            "global_repr": global_repr,
        }
