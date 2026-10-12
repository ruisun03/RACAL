from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class RecurrentTemporalEncoder(nn.Module):
    """Shared recurrent encoder for each univariate node sequence."""

    def __init__(self, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0, cell_type: str = "gru") -> None:
        super().__init__()
        cell_type = cell_type.lower()
        if cell_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported recurrent cell type: {cell_type}")

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.hidden_dim = hidden_dim
        recurrent_cls = nn.GRU if cell_type == "gru" else nn.LSTM
        self.recurrent = recurrent_cls(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.node_projection = nn.Linear(hidden_dim, hidden_dim)
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, window_size, num_nodes = _validate_window_input(x)

        node_sequences = x.transpose(1, 2).contiguous().view(batch_size * num_nodes, window_size, 1)
        _, hidden = self.recurrent(node_sequences)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        node_repr = hidden[-1].view(batch_size, num_nodes, self.hidden_dim)
        node_repr = self.node_projection(node_repr)
        global_repr = self.global_projection(node_repr.mean(dim=1))
        return {
            "node_repr": node_repr,
            "global_repr": global_repr,
        }


class CausalConvBlock(nn.Module):
    """A residual causal 1D convolution block for short windows."""

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.depthwise(x)
        if self.padding > 0:
            hidden = hidden[..., : -self.padding]
        hidden = self.pointwise(hidden)
        hidden = self.dropout(self.activation(self.norm(hidden)))
        return hidden + residual


class TCNTemporalEncoder(nn.Module):
    """Lightweight temporal convolutional encoder shared by all node sequences.

    The encoder applies a few causal convolutions directly on each univariate
    node sequence. It intentionally avoids hidden-to-hidden convolutions so the
    encoder replacement ablation stays comparable in runtime to GRU/LSTM.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        branch_count = max(1, min(int(num_layers), 3))
        self.dilations = [2**idx for idx in range(branch_count)]
        self.paddings = [dilation * (kernel_size - 1) for dilation in self.dilations]
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
                for dilation, padding in zip(self.dilations, self.paddings)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.node_projection = nn.Linear(hidden_dim, hidden_dim)
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, window_size, num_nodes = _validate_window_input(x)

        node_sequences = x.transpose(1, 2).contiguous().view(batch_size * num_nodes, window_size, 1)
        sequence_channels = node_sequences.transpose(1, 2)
        branch_outputs = []
        for branch, padding in zip(self.branches, self.paddings):
            hidden = branch(sequence_channels)
            if padding > 0:
                hidden = hidden[..., :-padding]
            branch_outputs.append(hidden[..., -1])

        node_repr = torch.stack(branch_outputs, dim=0).mean(dim=0)
        node_repr = node_repr.view(batch_size, num_nodes, self.hidden_dim)
        node_repr = self.node_projection(self.dropout(self.activation(self.norm(node_repr))))
        global_repr = self.global_projection(node_repr.mean(dim=1))
        return {
            "node_repr": node_repr,
            "global_repr": global_repr,
        }


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for transformer windows."""

    def __init__(self, hidden_dim: int, max_len: int) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / float(hidden_dim))
        )
        encoding = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class TransformerTemporalEncoder(nn.Module):
    """Transformer encoder shared by all node sequences."""

    def __init__(
        self,
        hidden_dim: int,
        window_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % nhead != 0:
            nhead = 1
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(1, hidden_dim)
        self.position = SinusoidalPositionalEncoding(hidden_dim=hidden_dim, max_len=window_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))
        self.node_projection = nn.Linear(hidden_dim, hidden_dim)
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, window_size, num_nodes = _validate_window_input(x)

        node_sequences = x.transpose(1, 2).contiguous().view(batch_size * num_nodes, window_size, 1)
        hidden = self.input_projection(node_sequences)
        hidden = self.position(hidden)
        hidden = self.encoder(hidden)

        node_repr = hidden[:, -1].view(batch_size, num_nodes, self.hidden_dim)
        node_repr = self.node_projection(node_repr)
        global_repr = self.global_projection(node_repr.mean(dim=1))
        return {
            "node_repr": node_repr,
            "global_repr": global_repr,
        }


def _validate_window_input(x: torch.Tensor) -> tuple[int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"Expected input of shape [B, T, N], got {tuple(x.shape)}")
    batch_size, window_size, num_nodes = x.shape
    return batch_size, window_size, num_nodes


class TemporalEncoder(nn.Module):
    """多变量滑动窗口的可替换时序编码器。

    输入 ``x`` 的形状为 ``[B, T, N]``，输出节点级表示 ``[B, N, D]`` 和窗口级表示
    ``[B, D]``。默认 ``encoder_type='gru'``，保持原始模型行为。
    """

    def __init__(
        self,
        hidden_dim: int,
        gru_layers: int = 1,
        dropout: float = 0.0,
        encoder_type: str = "gru",
        window_size: int = 10,
    ) -> None:
        super().__init__()
        encoder_type = encoder_type.lower()
        self.encoder_type = encoder_type
        if encoder_type == "gru":
            self.encoder = RecurrentTemporalEncoder(
                hidden_dim=hidden_dim,
                num_layers=gru_layers,
                dropout=dropout,
                cell_type="gru",
            )
        elif encoder_type == "lstm":
            self.encoder = RecurrentTemporalEncoder(
                hidden_dim=hidden_dim,
                num_layers=gru_layers,
                dropout=dropout,
                cell_type="lstm",
            )
        elif encoder_type == "tcn":
            self.encoder = TCNTemporalEncoder(
                hidden_dim=hidden_dim,
                num_layers=max(1, gru_layers + 1),
                dropout=dropout,
            )
        elif encoder_type == "transformer":
            self.encoder = TransformerTemporalEncoder(
                hidden_dim=hidden_dim,
                window_size=window_size,
                num_layers=max(1, gru_layers),
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unsupported temporal encoder: {encoder_type}. "
                "Expected one of ['gru', 'lstm', 'tcn', 'transformer']."
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder(x)
