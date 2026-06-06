import copy
import logging
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer

from model.transformer import TransformerLM
from training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    update_best_loss_and_save,
    validate_checkpoint,
)


def test_save_checkpoint_creates_file_with_expected_keys(
    tmp_path: Path,
    tiny_cfg: DictConfig,
    tiny_model: TransformerLM,
    tiny_optimizer: Optimizer,
    logger: logging.Logger,
) -> None:
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=path,
        model=tiny_model,
        optimizer=tiny_optimizer,
        epoch=0,
        cfg=tiny_cfg,
        train_metrics={"loss": 1.0},
        val_metrics={"loss": 1.0},
        best_val_loss=1.0,
        logger=logger,
    )

    checkpoint = torch.load(f=path, map_location="cpu", weights_only=False)

    assert path.exists()
    assert checkpoint["epoch"] == 0
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "rng_state" in checkpoint


def test_load_checkpoint_restores_model_state_and_returns_next_epoch(
    tmp_path: Path,
    tiny_cfg: DictConfig,
    tiny_model: TransformerLM,
    tiny_optimizer: Optimizer,
    logger: logging.Logger,
) -> None:
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=path,
        model=tiny_model,
        optimizer=tiny_optimizer,
        epoch=2,
        cfg=tiny_cfg,
        train_metrics={"loss": 1.0},
        val_metrics={"loss": 0.8},
        best_val_loss=0.8,
        logger=logger,
    )

    saved_state = copy.deepcopy(tiny_model.state_dict())

    with torch.no_grad():
        for param in tiny_model.parameters():
            param.add_(1.0)

    next_epoch, best_val_loss = load_checkpoint(
        path=path,
        model=tiny_model,
        optimizer=tiny_optimizer,
        device=torch.device("cpu"),
        logger=logger,
    )

    assert next_epoch == 3
    assert best_val_loss == 0.8

    for name, tensor in tiny_model.state_dict().items():
        assert torch.equal(tensor, saved_state[name])


def test_load_checkpoint_raises_for_missing_file(
    tiny_model: TransformerLM,
    tiny_optimizer: Optimizer,
    logger: logging.Logger,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            path="missing-checkpoint.pt",
            model=tiny_model,
            optimizer=tiny_optimizer,
            device=torch.device("cpu"),
            logger=logger,
        )


def test_validate_checkpoint_rejects_missing_required_keys(tmp_path: Path) -> None:
    checkpoint = {"epoch": 0}

    with pytest.raises(ValueError, match="missing keys"):
        validate_checkpoint(checkpoint=checkpoint, path=tmp_path / "bad.pt")


def test_validate_checkpoint_rejects_non_dict(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="expected dict"):
        validate_checkpoint(checkpoint=["not", "a", "dict"], path=tmp_path / "bad.pt")


def test_update_best_loss_and_saves_writes_best_latest_interval(
    tmp_path: Path,
    tiny_cfg: DictConfig,
    tiny_model: TransformerLM,
    tiny_optimizer: Optimizer,
    logger: logging.Logger,
) -> None:
    tiny_cfg.checkpoint.dir = str(tmp_path)
    tiny_cfg.checkpoint.checkpoint_interval = 1
    tiny_cfg.checkpoint.save_best = True
    tiny_cfg.checkpoint.save_latest = True

    best_val_loss = update_best_loss_and_save(
        cfg=tiny_cfg,
        model=tiny_model,
        optimizer=tiny_optimizer,
        epoch=0,
        train_metrics={"loss": 1.0},
        val_metrics={"loss": 0.9},
        best_val_loss=float("inf"),
        logger=logger,
    )

    assert best_val_loss == 0.9
    assert (tmp_path / "epoch_0001.pt").exists()
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "best.pt").exists()


def test_update_best_loss_and_save_does_not_write_when_disabled(
    tmp_path: Path,
    tiny_cfg: DictConfig,
    tiny_model: TransformerLM,
    tiny_optimizer: Optimizer,
    logger: logging.Logger,
) -> None:
    tiny_cfg.checkpoint.enabled = False
    tiny_cfg.checkpoint.dir = str(tmp_path)

    best_val_loss = update_best_loss_and_save(
        cfg=tiny_cfg,
        model=tiny_model,
        optimizer=tiny_optimizer,
        epoch=0,
        train_metrics={"loss": 1.0},
        val_metrics={"loss": 0.9},
        best_val_loss=float("inf"),
        logger=logger,
    )

    assert best_val_loss == 0.9
    assert not any(tmp_path.iterdir())
