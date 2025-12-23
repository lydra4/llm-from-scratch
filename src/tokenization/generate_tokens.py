import logging
import os
from typing import (
    List,
    Optional,
)

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

    def _convert_text_to_bytes(self, text: str, encoding: str) -> List[int]:
        byte_data = text.encode(encoding)
        tokens = list(byte_data)
        return tokens
