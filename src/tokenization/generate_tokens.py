import logging
import os
from typing import Optional

from omegaconf import DictConfig


class GenerateTokens:
    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger

        train_path = os.path.join(self.cfg.data_path, "train", "train.txt")
        with open(file=train_path, mode="r", encoding="utf-8") as f:
            self.train_text = f.read()
