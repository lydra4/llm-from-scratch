import logging

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from dataset.dataloaders import create_dataloaders
from training.trainer import train_epoch, validate_epoch
from utils.general_utils import setup_logging
from utils.setup import setup_model_and_components


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging()

    device = torch.device(device=cfg.hardware.device)
    logger.info(f"Training model on {device}.")

    train_loader, val_loader = create_dataloaders(cfg=cfg, logger=logger)

    model, optimizer, criterion = setup_model_and_components(
        cfg=cfg,
        device=device,
        logger=logger,
    )

    logger.info("Training....")
    for epoch in tqdm(iterable=range(cfg.model.epochs)):
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cfg=cfg,
            logger=logger,
            epoch=epoch,
        )

        val_loss = validate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
            logger=logger,
            epoch=epoch,
        )

        print(train_loss, val_loss)


if __name__ == "__main__":
    main()
