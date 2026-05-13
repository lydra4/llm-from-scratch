import logging

import torch
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    device: torch.device,
    cfg: DictConfig,
    logger: logging.Logger,
    epoch: int,
) -> float:
    model.train()
    total_train_loss = 0

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits.view(-1, cfg.model.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    logger.info(f"Epoch {epoch} | Average Training Loss: {avg_train_loss:.4f}")

    return avg_train_loss


def validate_epoch(
    model: Module,
    val_loader: DataLoader,
    criterion: Module,
    device: torch.device,
    cfg: DictConfig,
    logger: logging.Logger,
    epoch: int,
) -> float:
    model.eval()
    total_val_loss = 0

    logger.info(f"Scoring on validation for epoch: {epoch:.4f}")
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)

            val_logits = model(x_val)
            val_loss = criterion(
                val_logits.view(-1, cfg.model.vocab_size), y_val.view(-1)
            )
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    logger.info(f"Epoch {epoch} | Val loss: {avg_val_loss:.4f}")

    return avg_val_loss
