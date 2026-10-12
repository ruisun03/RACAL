from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from utils.graph_utils import adjacency_to_tensor


class NodeMechanismMLP(nn.Module):
    """因果残差模块内部的节点级小预测器。

    每个节点都有自己独立的一套 MLP，因为即便共享同一张图，不同节点的
    局部生成机制也可能不同。
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """根据历史特征预测该节点窗口最后一个时刻的数值。"""

        return self.network(features).squeeze(-1)


class CausalResidualModule(nn.Module):
    """衡量每个窗口与“图条件下局部机制”匹配得有多好。

    对于节点 j，这个模块会利用两部分信息去预测窗口最后一个时刻：
    - 节点自身在前 ``T-1`` 步的历史
    - 根据邻接矩阵聚合得到的父节点历史

    节点级绝对预测误差会形成 ``node_residuals``，再对节点维求均值得到
    窗口级的 ``window_causal_score``。
    """

    def __init__(
        self,
        num_nodes: int,
        window_size: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = 2 * (window_size - 1)
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.predictors = nn.ModuleList(
            [NodeMechanismMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_nodes)]
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测每个节点最后一步，并返回基于残差的打分结果。"""

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape [B, T, N], got {tuple(x.shape)}")
        if x.shape[1] != self.window_size:
            raise ValueError(f"Expected window size {self.window_size}, got {x.shape[1]}")
        if x.shape[2] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[2]}")

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)

        # 把窗口最后一个时刻作为目标值，前 ``T-1`` 个时刻作为机制输入。
        history = x[:, :-1, :]
        targets = x[:, -1, :]

        # ``parent_history[:, :, j]`` 聚合所有指向节点 j 的父节点历史。
        # 再除以入度，是为了让“父节点很多的节点”和“父节点很少的节点”保持
        # 相近的数值尺度。
        parent_history = torch.einsum("ij,bti->btj", adjacency_t, history)
        in_degree = torch.clamp(adjacency_t.sum(dim=0), min=1.0).view(1, 1, -1)
        parent_history = parent_history / in_degree

        predictions = []
        for node_idx, predictor in enumerate(self.predictors):
            node_self_history = history[:, :, node_idx]
            node_parent_history = parent_history[:, :, node_idx]
            # 每个节点的局部机制只看两部分：自身历史 + 图定义下的父节点摘要。
            features = torch.cat([node_self_history, node_parent_history], dim=-1)
            predictions.append(predictor(features))

        predicted_last = torch.stack(predictions, dim=-1)
        residuals = torch.abs(predicted_last - targets)
        # 项目当前把“节点绝对残差均值”直接当作窗口级因果异常强度。
        window_causal_score = residuals.mean(dim=-1)
        # 这里也会额外算一个 MSE，方便观察机制预测质量；但当前训练目标
        # 实际优化的是 ``window_causal_score.mean()``，不是这个 MSE。
        mse_loss = torch.mean((predicted_last - targets) ** 2)

        return {
            "predictions": predicted_last,
            "node_residuals": residuals,
            "window_causal_score": window_causal_score,
            "mse_loss": mse_loss,
        }
