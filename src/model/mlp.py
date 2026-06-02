from typing import Literal, cast

import torch
from omegaconf import DictConfig
from torch import nn

from model.modules.layers import GPTLinear


class FeedForward(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d_model = self.cfg.model.d_model
        hidden_dim = self.cfg.model.get("mlp_hidden_dim", 4 * d_model)
        dropout = self.cfg.model.dropout
        bias = self.cfg.model.get("bias", True)
        approximate_cfg = self.cfg.model.get("approximate", "tanh")
        if approximate_cfg not in {"none", "tanh"}:
            raise ValueError(f"Invalid GELU approximate value: {approximate_cfg}")

        approximate = cast(Literal["none", "tanh"], approximate_cfg)

        self.c_fc = nn.Linear(
            in_features=d_model,
            out_features=hidden_dim,
            bias=bias,
        )
        self.gelu = nn.GELU(approximate=approximate)
        self.c_proj = GPTLinear(
            in_features=hidden_dim,
            out_features=d_model,
            bias=bias,
            scale_init=True,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return self.dropout(x)
