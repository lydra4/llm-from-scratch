import logging
from typing import Optional

from omegaconf import DictConfig
from torch import nn


class TransformerLM(nn.Module):
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger]) -> None:
        super().__init__()
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
