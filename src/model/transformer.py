import logging
from typing import Optional

from omegaconf import DictConfig


class TransformerLM:
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger,]) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
