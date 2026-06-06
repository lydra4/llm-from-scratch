import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dataset.dataloaders import create_dataloaders
from dataset.token_dataset import TokenDataset


def test_token_dataset_returns_shifted_language_modelling_pairs(
    token_array_path: str | os.PathLike,
    logger: logging.Logger,
) -> None:
    dataset = TokenDataset(
        data_path=token_array_path,
        context_window=4,
        logger=logger,
    )

    x, y = dataset[0]

    assert len(dataset) == 16
    assert torch.equal(x, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(y, torch.tensor([1, 2, 3, 4]))


def test_token_dataset_rejects_out_of_bounds_index(
    token_array_path: str | os.PathLike,
    logger: logging.Logger,
) -> None:
    dataset = TokenDataset(data_path=token_array_path, context_window=4, logger=logger)

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_token_dataset_rejects_too_short_data(
    tmp_path: Path,
    logger: logging.Logger,
) -> None:
    path = tmp_path / "short.npy"
    np.save(
        file=path,
        arr=np.array([1, 2, 3, 4], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="Invalid numpy file"):
        TokenDataset(data_path=str(path), context_window=4, logger=logger)


def test_token_dataset_rejects_non_integer_data(
    tmp_path: Path,
    logger: logging.Logger,
) -> None:
    path = tmp_path / "floats.npy"
    np.save(file=path, arr=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    with pytest.raises(ValueError, match="Invalid numpy file"):
        TokenDataset(data_path=str(path), context_window=4, logger=logger)


def test_create_dataloaders_returns_expected_batch_shapes(
    tmp_path: Path,
    logger: logging.Logger,
) -> None:
    train_path = tmp_path / "train.npy"
    val_path = tmp_path / "val.npy"
    np.save(file=train_path, arr=np.arange(20, dtype=np.int64))
    np.save(file=val_path, arr=np.arange(20, dtype=np.int64))

    cfg = OmegaConf.create(
        {
            "dataset": {"context_window": 4},
            "train": {
                "data_path": str(train_path),
                "loader": {"batch_size": 2, "shuffle": False},
            },
            "val": {
                "data_path": str(val_path),
                "loader": {"batch_size": 3, "shuffle": False},
            },
        }
    )

    train_loader, val_loader = create_dataloaders(cfg=cfg, logger=logger)
    train_x, train_y = next(iter(train_loader))
    val_x, val_y = next(iter(val_loader))

    assert train_x.shape == torch.Size([2, 4])
    assert train_y.shape == torch.Size([2, 4])
    assert val_x.shape == torch.Size([3, 4])
    assert val_y.shape == torch.Size([3, 4])
