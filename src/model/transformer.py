import logging

import torch
from omegaconf import DictConfig
from torch import nn

from model.modules.block import TransformerBlock
from model.modules.embeddings import TransformerEmbeddings
from model.modules.layers import GPTLinear


class TransformerLM(nn.Module):
    def __init__(self, cfg: DictConfig, logger: logging.Logger | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        self.embeddings = TransformerEmbeddings(cfg=self.cfg, logger=self.logger)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg=self.cfg) for _ in range(self.cfg.model.n_layers)]
        )

        self.ln_f = nn.LayerNorm(normalized_shape=self.cfg.model.d_model)
        self.lm_head = nn.Linear(
            in_features=self.cfg.model.d_model,
            out_features=self.cfg.model.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

        if self.cfg.model.get("tie_weights", True):
            self.lm_head.weight = self.embeddings.token_embeddings.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02

            if isinstance(module, GPTLinear) and module.scale_init:
                std *= (2 * self.cfg.model.n_layers) ** -0.5

            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        return self.lm_head(x)

    def num_parameters(self, non_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())

        if non_embedding:
            total -= self.embeddings.token_embeddings.weight.numel()

        return total
