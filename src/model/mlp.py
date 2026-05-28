import torch
from omegaconf import DictConfig
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.d_model = self.cfg.model.d_model
        hidden_dim = 4 * self.d_model
        dropout = self.cfg.model.dropout
        self.net = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
