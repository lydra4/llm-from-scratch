import logging
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn


class TransformerEmbeddings(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[logging.Logger],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        self.d_model = self.cfg.d_model
        self.token_embeddings = nn.Embedding(
            num_embeddings=self.cfg.model.vocab_size,
            embedding_dim=self.cfg.model.d_model,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=self.cfg.model.context_window,
            embedding_dim=self.cfg.model.d_model,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.size()
        device = idx.device

        pos = torch.arange(start=0, end=t, dtype=torch.long, device=device)

        token_embeddings = self.token_embeddings(idx)
        position_embeddings = self.position_embeddings(pos)

        return token_embeddings + position_embeddings
