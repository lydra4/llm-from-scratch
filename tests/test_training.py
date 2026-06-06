import logging
import math

import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from model.transformer import TransformerLM
from training.trainer import train_epoch, validate_epoch


def test_train_epoch_returns_finite_metrics(
    tiny_cfg: DictConfig,
    tiny_model: TransformerLM,
    tiny_optimizer: Optimizer,
    tiny_train_loader: DataLoader,
    logger: logging.Logger,
) -> None:
    criterion = torch.nn.CrossEntropyLoss()

    metrics = train_epoch(
        model=tiny_model,
        train_loader=tiny_train_loader,
        optimizer=tiny_optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        cfg=tiny_cfg,
        logger=logger,
        epoch=0,
    )

    assert math.isfinite(metrics["loss"])
    assert metrics["num_batches"] == len(tiny_train_loader)
    assert metrics["samples_per_sec"] >= 0
    assert metrics["tokens_per_sec"] >= 0
    assert metrics["epoch_time_sec"] >= 0


def test_validate_epoch_returns_finite_metrics(
    tiny_cfg: DictConfig,
    tiny_model: TransformerLM,
    tiny_val_loader: DataLoader,
    logger: logging.Logger,
) -> None:
    criterion = torch.nn.CrossEntropyLoss()

    metrics = validate_epoch(
        model=tiny_model,
        val_loader=tiny_val_loader,
        criterion=criterion,
        device=torch.device("cpu"),
        cfg=tiny_cfg,
        logger=logger,
        epoch=0,
    )

    assert math.isfinite(metrics["loss"])
    assert metrics["num_batches"] == len(tiny_val_loader)


def test_train_epoch_rejects_invalid_vocab_size(
    tiny_cfg: DictConfig,
    tiny_model: TransformerLM,
    tiny_optimizer: Optimizer,
    tiny_train_loader: DataLoader,
    logger: logging.Logger,
) -> None:
    tiny_cfg.model.vocab_size = 0

    try:
        train_epoch(
            model=tiny_model,
            train_loader=tiny_train_loader,
            optimizer=tiny_optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            cfg=tiny_cfg,
            logger=logger,
            epoch=0,
        )

    except ValueError as e:
        assert "vocab_size must be > 0" in str(e)

    else:
        raise AssertionError("Expected train_epoch to reject vocab_size <= 0")
