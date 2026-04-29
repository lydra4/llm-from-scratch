import logging

import hydra
from omegaconf import DictConfig

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

    token_dataset = TokenDataset(cfg=cfg, logger=logger)
    print(token_dataset)


if __name__ == "main":
    main()
