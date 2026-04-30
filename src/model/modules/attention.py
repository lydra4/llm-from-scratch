import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[logging.Logger],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        self.d_model = self.cfg.model.d_model
        self.n_heads = self.cfg.model.n_heads
        self.head_dim = self.d_model // self.n_heads

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

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

        context_window = self.cfg.model.context_window
        mask = torch.tril(input=torch.ones(size=(context_window, context_window)))
        self.register_buffer(
            name="bias",
            tensor=mask.view(1, 1, context_window, context_window),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = F.softmax(input=attn, dim=-1)

        y = attn @ v
        y = y.transpose(1, 2).contiguouse.view(b, t, c)

        return self.c_proj(y)
