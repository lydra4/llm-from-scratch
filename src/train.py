import logging

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset.token_dataset import TokenDataset
from utils.general_utils import setup_logging


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging()

    train_dataset = TokenDataset(
        data_path=cfg.train.data_path,
        context_window=cfg.dataset.context_window,
        logger=logger,
    )
    train_loader = DataLoader(dataset=train_dataset, **cfg.train.loader)

    val_dataset = TokenDataset(
        data_path=cfg.val.data_path,
        context_window=cfg.dataset.context_window,
        logger=logger,
    )
    val_loader = DataLoader(dataset=val_dataset, **cfg.val.loader)
    print(train_loader, val_loader)


if __name__ == "__main__":
    main()
