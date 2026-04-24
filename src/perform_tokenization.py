import logging

import hydra
from omegaconf import DictConfig

from tokenization.generate_tokens import GenerateTokens
from utils.general_utils import setup_logging


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="tokenization.yaml",
)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging()

    token_generator = GenerateTokens(cfg=cfg, logger=logger)
    token_generator.tokenize_text()


if __name__ == "__main__":
    main()
