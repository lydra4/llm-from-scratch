import logging

import hydra
from omegaconf import DictConfig

from tokenizer.bpe_tokenizer import BPETokenizer
from utils.general_utils import setup_logging


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="encoding.yaml",
)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging()

    bpe_tokenizer = BPETokenizer(cfg=cfg, logger=logger)
    bpe_tokenizer.encode_all_text()


if __name__ == "__main__":
    main()
