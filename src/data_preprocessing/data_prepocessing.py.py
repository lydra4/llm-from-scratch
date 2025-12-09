import logging
from typing import Optional

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from omegaconf import DictConfig


class DataPreprocessing:
    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def _extract_epub_text(self, path: str) -> str:
        book = epub.read_epub(name=path)
        texts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                if item.get_name() not in self.cfg.exclude_files:
                    soup = BeautifulSoup(item.get_body_content(), "html.parser")
                    texts.append(soup.get_text(separator="\n"))

        return "\n".join(texts)
