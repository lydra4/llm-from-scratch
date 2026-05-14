import logging
from typing import Optional

import torch
from modules.attention import CausalSelfAttention
from modules.embeddings import TransformerEmbeddings
from omegaconf import DictConfig
from torch import nn

from model.modules.block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger]) -> None:
        super().__init__()
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        self.embeddings = TransformerEmbeddings(cfg=self.cfg, logger=self.logger)
        self.attention = CausalSelfAttention(cfg=self.cfg, logger=self.logger)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg=self.cfg) for _ in range(self.n_layers)]
        )

        self.ln_f = nn.LayerNorm(normalized_shape=self.d_model)
        self.lm_head = nn.Linear(
            in_features=self.d_model,
            out_features=self.vocab_size,
            bias=False,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits
