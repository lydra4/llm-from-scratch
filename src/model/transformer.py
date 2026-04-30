import logging
from typing import Optional

import torch
from modules.attention import CausalSelfAttention
from modules.embeddings import TransformerEmbeddings
from omegaconf import DictConfig
from torch import nn


class TransformerLM(nn.Module):
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger]) -> None:
        super().__init__()
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        self.embeddings = TransformerEmbeddings(cfg=self.cfg, logger=self.logger)
        self.attention = CausalSelfAttention(cfg=self.cfg, logger=self.logger)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(idx)
        x = self.attention(x)
        return x
