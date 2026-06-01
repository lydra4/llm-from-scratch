import logging
import math

import torch
from omegaconf import DictConfig
from torch import nn


class TransformerEmbeddings(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        self.token_embeddings = nn.Embedding(
            num_embeddings=self.cfg.model.vocab_size,
            embedding_dim=self.cfg.model.d_model,
        )
        pe = self._build_positional_embeddings(
            context_window=self.cfg.dataset.context_window,
            d_model=self.cfg.model.d_model,
        )
        self.pe: torch.Tensor
        self.register_buffer(name="pe", tensor=pe)

        self.dropout = nn.Dropout(p=self.cfg.model.dropout)

    def _build_positional_embeddings(
        self,
        context_window: int,
        d_model: int,
    ) -> torch.Tensor:
        pe = torch.zeros(context_window, d_model)

        position = torch.arange(
            start=0,
            end=context_window,
            dtype=torch.float,
        ).unsqueeze(1)

        div_term = torch.exp(
            input=torch.arange(start=0, end=d_model, step=2).float()
            * (-math.log(10_000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.size()

        token_embeddings = self.token_embeddings(idx)
        positional_embeddings = self.pe[:t, :].unsqueeze(0)

        return self.dropout(token_embeddings + positional_embeddings)
