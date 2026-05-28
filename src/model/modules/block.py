import torch
from omegaconf import DictConfig
from torch import nn

from model.mlp import FeedForward
from model.modules.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.ln_1 = nn.LayerNorm(normalized_shape=self.cfg.model.d_model)
        self.ln_2 = nn.LayerNorm(normalized_shape=self.cfg.model.d_model)

        self.attention = CausalSelfAttention(cfg=cfg)
        self.mlp = FeedForward(cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x
