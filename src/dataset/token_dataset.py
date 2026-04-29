import logging
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[logging.Logger],
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.data = np.load(file=self.cfg.npy_file_path, mmap_mode="r")

    def __len__(self) -> int:
        return self.cfg.total_tokens - self.cfg.context_window

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.cfg.context_window + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y
