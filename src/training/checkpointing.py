import logging
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module
from torch.optim import Optimizer


def save_checkpoint(
    path: str | os.PathLike,
    model: Module,
    optimizer: Optimizer,
    epoch: int,
    cfg: DictConfig,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
    logger: logging.Logger,
) -> None:
    checkpoint_dir = os.path.dirname(p=path)

    if checkpoint_dir:
        os.makedirs(name=checkpoint_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg=cfg, resolve=True),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }

    tmp_path = f"{path}.tmp"

    try:
        torch.save(obj=state, f=tmp_path)
        os.replace(src=tmp_path, dst=path)
    except Exception:
        if os.path.exists(path=tmp_path):
            os.remove(tmp_path)
        raise

    logger.info(f"Saved checkpoint: {path}.")


def load_checkpoint(
    path: str | os.PathLike,
    model: Module,
    optimizer: Optimizer,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, float]:
    if not os.path.exists(path=path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(
            f=path,
            map_location=device,
            weights_only=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read checkpoint at {path}") from e

    checkpoint = validate_checkpoint(checkpoint=checkpoint, path=path)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load model state from checkpoint at {path}."
            "This usually means the model architecture or config changed "
            "since the checkpoint was created"
        ) from e

    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except ValueError as e:
        raise RuntimeError(
            f"Failed to load optimizer state dict from checkpoint at {path}."
            "This usually means the optimizer configuration changed "
            "since the checkpoint was created"
        ) from e

    rng_state = checkpoint.get("rng_state")

    if rng_state is not None:
        torch_rng_state = rng_state.get("torch")
        cuda_rng_state = rng_state.get("cuda")

        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state)

        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)

    completed_epoch = int(checkpoint["epoch"])
    next_epoch = completed_epoch + 1
    best_val_loss = float(checkpoint["best_val_loss"])

    logger.info(
        f"Loaded checkpoint: {path}. "
        f"Completed epoch {completed_epoch + 1};"
        f"resuming from epoch {next_epoch + 1}."
    )

    return next_epoch, best_val_loss


def resume_checkpoint(
    cfg: DictConfig,
    model: Module,
    optimizer: Optimizer,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, float]:
    start_epoch = 0
    best_val_loss = float("inf")

    if cfg.checkpoint.resume_path:
        start_epoch, best_val_loss = load_checkpoint(
            path=cfg.checkpoint.resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

    return start_epoch, best_val_loss


def update_best_loss_and_save(
    cfg: DictConfig,
    model: Module,
    optimizer: Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
    logger: logging.Logger,
) -> float:
    val_loss = val_metrics["loss"]
    is_best = val_loss < best_val_loss

    if is_best:
        logger.info(f"Validation loss improved: {best_val_loss:.4f} -> {val_loss:.4f}")
        best_val_loss = val_loss

    if not cfg.checkpoint.enabled:
        logger.debug("Checkpointing is disabled.")
        return best_val_loss

    checkpoint_dir = cfg.checkpoint.dir
    checkpoint_interval = cfg.checkpoint.checkpoint_interval
    save_best = cfg.checkpoint.save_best
    save_latest = cfg.checkpoint.get("save_latest", True)

    should_save_interval = (
        checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0
    )
    should_save_best = save_best and is_best
    should_save_latest = save_latest and should_save_interval

    if should_save_interval:
        logger.info(f"Saving interval checkpoint for each epoch {epoch + 1}.")
        save_checkpoint(
            path=os.path.join(checkpoint_dir, f"epoch_{epoch + 1:04d}.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val_loss=best_val_loss,
            logger=logger,
        )

    if should_save_latest:
        logger.info(f"Saving latest checkpoint for epoch {epoch + 1}")
        save_checkpoint(
            path=os.path.join(checkpoint_dir, "latest.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val_loss=best_val_loss,
            logger=logger,
        )

    if should_save_best:
        logger.info(f"Saving best checkpoint for epoch {epoch + 1}")
        save_checkpoint(
            path=os.path.join(checkpoint_dir, "best.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val_loss=best_val_loss,
            logger=logger,
        )

    return best_val_loss


def validate_checkpoint(checkpoint: object, path: str | os.PathLike) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Invalid checkpoint at {path}: expected dict, "
            f"got {type(checkpoint).__name__}"
        )

    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "config",
        "train_metrics",
        "val_metrics",
        "best_val_loss",
    }

    missing_keys = required_keys - checkpoint.keys()

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Invalid checkpoint at {path}: missing keys: {missing}")

    return checkpoint
