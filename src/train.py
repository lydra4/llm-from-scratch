import logging

import hydra
from omegaconf import DictConfig

from training.runner import run_training
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

    run_training(cfg=cfg, logger=logger)


if __name__ == "__main__":
    main()
