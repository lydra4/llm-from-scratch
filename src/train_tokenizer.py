import logging

import hydra
from omegaconf import DictConfig

from tokenizer.bpe_builder import BytePairEncoder
from utils.general_utils import setup_logging


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="bpe.yaml",
)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging()

    bytepair_encoder = BytePairEncoder(cfg=cfg, logger=logger)
    bytepair_encoder.tokenize_text()


if __name__ == "__main__":
    main()
