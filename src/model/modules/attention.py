import logging

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        self.d_model = self.cfg.model.d_model
        self.n_heads = self.cfg.model.n_heads
        self.head_dim = self.d_model // self.n_heads

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.dropout = self.cfg.model.get("dropout", 0.1)
        self.c_attn = nn.Linear(
            in_features=self.d_model,
            out_features=(3 * self.d_model),
            bias=False,
        )
        self.c_proj = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(b, t, c)

        return self.c_proj(y)
