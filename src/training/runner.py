import logging

import torch
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataloaders import create_dataloaders
from tracking.mlflow_tracker import log_epoch_metrics, log_model, start_mlflow_run
from training.checkpointing import (
    resume_checkpoint,
    update_best_loss_and_save,
)
from training.trainer import train_epoch, validate_epoch
from utils.setup import setup_model_and_components


def run_epoch(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    device: torch.device,
    cfg: DictConfig,
    logger: logging.Logger,
    epoch: int,
    best_val_loss: float,
) -> float:
    train_metrics = train_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        cfg=cfg,
        logger=logger,
        epoch=epoch,
    )

    val_metrics = validate_epoch(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        cfg=cfg,
        logger=logger,
        epoch=epoch,
    )

    log_epoch_metrics(train_metrics=train_metrics, val_metrics=val_metrics, epoch=epoch)

    best_val_loss = update_best_loss_and_save(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        best_val_loss=best_val_loss,
        logger=logger,
    )

    log_model(cfg=cfg, model=model, epoch=epoch)

    return best_val_loss


def run_training(cfg: DictConfig, logger: logging.Logger) -> None:
    device = torch.device(device=cfg.hardware.device)
    logger.info(f"Training model on {device}.")

    with start_mlflow_run(cfg=cfg):
        train_loader, val_loader = create_dataloaders(cfg=cfg, logger=logger)

        model, optimizer, criterion = setup_model_and_components(
            cfg=cfg,
            device=device,
            logger=logger,
        )

        start_epoch, best_val_loss = resume_checkpoint(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

        logger.info("Training...")
        for epoch in tqdm(iterable=range(start_epoch, cfg.model.epochs)):
            best_val_loss = run_epoch(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                cfg=cfg,
                logger=logger,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )
