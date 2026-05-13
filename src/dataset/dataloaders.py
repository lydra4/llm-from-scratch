import logging

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset.token_dataset import TokenDataset


def create_dataloaders(
    cfg: DictConfig,
    logger: logging.Logger,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TokenDataset(
        data_path=cfg.train.data_path,
        context_window=cfg.dataset.context_window,
        logger=logger,
    )
    train_loader = DataLoader(train_dataset, **cfg.train.loader)

    val_dataset = TokenDataset(
        data_path=cfg.val.data_path,
        context_window=cfg.dataset.context_window,
        logger=logger,
    )

    val_loader = DataLoader(dataset=val_dataset, **cfg.val.loader)

    return train_loader, val_loader
