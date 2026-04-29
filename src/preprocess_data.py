import logging

import hydra
from omegaconf import DictConfig

from preprocessing.preprocessor import DataPreprocessor
from utils.general_utils import setup_logging


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="preprocessing.yaml",
)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging()

    data_preprocessing = DataPreprocessor(cfg=cfg, logger=logger)
    data_preprocessing.perform_processing()


if __name__ == "__main__":
    main()
