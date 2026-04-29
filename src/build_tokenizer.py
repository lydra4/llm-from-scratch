import logging

import hydra
from omegaconf import DictConfig

from tokenizer.bpe_builder import BPEBuilder
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

    bytepair_builder = BPEBuilder(cfg=cfg, logger=logger)
    bytepair_builder.tokenize_text()


if __name__ == "__main__":
    main()
