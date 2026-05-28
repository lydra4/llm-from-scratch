import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        context_window: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.data_path = data_path
        self.context_window = context_window
        self.logger = logger or logging.getLogger(__name__)

        self._validate_inputs()

        try:
            self.data = np.load(file=self.data_path, mmap_mode="r")
            self._validate_data()
        except FileNotFoundError as e:
            self.logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}") from e
        except ValueError as e:
            self.logger.error(f"Invalid numpy file: {self.data_path}")
            raise ValueError(f"Invalid numpy file: {self.data_path}") from e

    def _validate_inputs(self) -> None:
        if not isinstance(self.data_path, str):
            raise TypeError(f"data_path must be str, got {type(self.data_path)}")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"data file does not exist: {self.data_path}")

        if not isinstance(self.context_window, int):
            raise TypeError(
                f"context_window must be int, got {type(self.context_window)}"
            )

        if self.context_window <= 0:
            raise ValueError(f"context_window must be > 0, got {self.context_window}")

    def _validate_data(self) -> None:
        if self.data.ndim != 1:
            raise ValueError(f"Data must be 1D array, got shape {self.data.shape}")

        if len(self.data) <= self.context_window:
            raise ValueError(
                f"Data length ({len(self.data)}) must be > context_window ({self.context_window})"
            )

        if not np.issubdtype(self.data.dtype, np.integer):
            raise ValueError(f"Data must contain integers, got dtype {self.data.dtype}")

        self.logger.info(
            f"Loaded data: shape={self.data.shape}, dtype={self.data.dtype}"
            f"context_window={self.context_window}"
        )

    def __len__(self) -> int:
        return len(self.data) - self.context_window

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self)}"
            )

        try:
            chunk = self.data[idx : idx + self.context_window + 1]
            x = torch.from_numpy(chunk[:-1].astype(np.int64))
            y = torch.from_numpy(chunk[1:].astype(np.int64))

            if torch.any(x < 0) or torch.any(y < 0):
                self.logger.warning(f"Negative token IDs at index {idx}")

            return x, y

        except Exception as e:
            self.logger.error(f"Error loading sample at index {idx}: {e}")
            raise RuntimeError(f"Error loading sample at index {idx}") from e
