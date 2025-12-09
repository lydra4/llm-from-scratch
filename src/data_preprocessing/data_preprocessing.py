import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from omegaconf import DictConfig
from tqdm import tqdm


class DataPreprocessing:
    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def _extract_clean_epub_books(self, epub_dict: Dict[str, List[str]]):
        for key, epub_list in epub_dict.items():
            self.logger.info(f"Processing {key} books.")
            texts = []
            for path in tqdm(iterable=epub_list):
                book = epub.read_epub(name=path)
                for idref, _ in book.spine:
                    item = book.get_item_with_id(uid=idref)

                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        if item.get_name() not in self.cfg.exclude_files:
                            soup = BeautifulSoup(item.get_body_content(), "html.parser")
                            raw_text = soup.get_text("\n")
                            clean_text = self._clean_text(text=raw_text)
                            texts.append(clean_text)

            return "\n".join(texts)

    def _clean_text(self, text: str) -> str:
        cleaned_text = unicodedata.normalize("NFKC", text)
        cleaned_text = cleaned_text.replace("“", '"').replace("”", '"')
        cleaned_text = cleaned_text.replace("‘", "'").replace("’", "'")
        cleaned_text = cleaned_text.replace("–", "-").replace("—", "-")
        cleaned_text = cleaned_text.replace("…", "...")
        cleaned_text = cleaned_text.replace("\u00a0", " ")
        cleaned_text = cleaned_text.replace("&nbsp;", " ").replace("&amp;", "&")

        cleaned_text = re.sub(r"\*{2,}", "<scene_break>", cleaned_text)
        cleaned_text = re.sub(r"_{2,}", "", cleaned_text)
        cleaned_text = re.sub(r"-{2,}", "", cleaned_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\n{1,}", "\n", cleaned_text)
        cleaned_text = re.sub(
            r"^\s*(page|chapter)\s*\d+\s*$",
            "",
            cleaned_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        cleaned_text = re.sub(r"-\n\s*", " ", cleaned_text)
        cleaned_text = re.sub(r"([!?.])\1{2,}", r"\1\1\1", cleaned_text)
        cleaned_text = re.sub(
            r"([^\w\s])\1{2,}",
            lambda m: m.group(1) * 2,
            cleaned_text,
        )
        return cleaned_text.strip()

    def _obtain_epub_list(self, path: str) -> Dict[str, List[str]]:
        walks = [
            (os.path.basename(dirpath), dirpath, filenames)
            for dirpath, _, filenames in os.walk(path)
        ]

        return {
            folder: [
                os.path.join(dirpath, file)
                for file in filenames
                if file.endswith(".epub")
            ]
            for folder, dirpath, filenames in walks
            if folder in ("GOT", "HP")
        }

    def perform_processing(self):
        epub_dict = self._obtain_epub_list(path=self.cfg.raw_epub_dir)
        texts = self._extract_clean_epub_books(epub_dict=epub_dict)
        print(texts)
