import logging
import os

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
    }

    torch.save(obj=state, f=path)
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: str | os.PathLike,
    model: Module,
    optimizer: Optimizer,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, float]:
    if not os.path.exists(path=path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        f=path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    completed_epoch = int(checkpoint["epoch"])
    next_epoch = completed_epoch + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    logger.info(f"Loaded checkpoint: {path}. Resuming from epoch {next_epoch}.")

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
        best_val_loss = val_loss

    if not cfg.checkpoint.enabled:
        return best_val_loss

    checkpoint_dir = cfg.checkpoint.dir
    should_save_interval = (
        cfg.checkpoint.checkpoint_interval > 0
        and (epoch + 1) % cfg.checkpoint.checkpoint_interval == 0
    )
    should_save_best = cfg.checkpoint.save_best and is_best

    if should_save_interval:
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

    if should_save_best:
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
