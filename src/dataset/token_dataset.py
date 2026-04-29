import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        context_window: int,
        logger: Optional[logging.Logger],
    ) -> None:
        self.data_path = data_path
        self.context_window = context_window
        self.logger = logger or logging.getLogger(__name__)
        self.data = np.load(file=self.data_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.data) - self.context_window

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_window + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y
