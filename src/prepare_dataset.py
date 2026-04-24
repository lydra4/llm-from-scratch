import logging

import hydra
from omegaconf import DictConfig

from utils.general_utils import setup_logging


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="prepare_dataset.yaml",
)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging()

    # tokenizer = Tokenizer(cfg=cfg, logger=logger)


if __name__ == "__main__":
    main()
