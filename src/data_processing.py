import logging
import os

import hydra
from omegaconf import DictConfig

from utils.general_utils import setup_logging


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="data_processing.yaml",
)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(),
            "conf",
            "logging.yaml",
        )
    )
