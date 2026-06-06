import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dataset.token_dataset import TokenDataset
from model.transformer import TransformerLM


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("tests")


@pytest.fixture
def tiny_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "dataset": {
                "context_window": 4,
            },
            "model": {
                "epochs": 2,
                "vocab_size": 32,
                "d_model": 8,
                "n_layers": 1,
                "n_heads": 2,
                "dropout": 0.0,
                "bias": True,
                "tie_weights": True,
                "mlp_hidden_dim": 16,
            },
            "checkpoint": {
                "enabled": True,
                "dir": "checkpoints",
                "checkpoint_interval": 1,
                "save_best": True,
                "save_latest": True,
                "resume_path": None,
            },
        }
    )


@pytest.fixture
def token_array_path(tmp_path: Path) -> str:
    path = tmp_path / "tokens.npy"
    np.save(file=path, arr=np.arange(20, dtype=np.int64))
    return str(path)


@pytest.fixture
def tiny_model(tiny_cfg: DictConfig, logger: logging.Logger) -> TransformerLM:
    torch.manual_seed(0)
    return TransformerLM(cfg=tiny_cfg, logger=logger)


@pytest.fixture
def tiny_optimizer(tiny_model: TransformerLM) -> Optimizer:
    return torch.optim.AdamW(params=tiny_model.parameters(), lr=1e-3)


@pytest.fixture
def tiny_train_loader(
    token_array_path: str,
    tiny_cfg: DictConfig,
    logger: logging.Logger,
) -> DataLoader:
    dataset = TokenDataset(
        data_path=token_array_path,
        context_window=tiny_cfg.dataset.context_window,
        logger=logger,
    )

    return DataLoader(dataset=dataset, batch_size=2, shuffle=False)


@pytest.fixture
def tiny_val_loader(
    token_array_path: str, tiny_cfg: DictConfig, logger: logging.Logger
) -> DataLoader:
    dataset = TokenDataset(
        data_path=token_array_path,
        context_window=tiny_cfg.dataset.context_window,
        logger=logger,
    )

    return DataLoader(dataset=dataset, batch_size=2, shuffle=False)
