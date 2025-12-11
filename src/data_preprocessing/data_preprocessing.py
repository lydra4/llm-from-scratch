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
        self.trans_table = str.maketrans(
            {
                "“": '"',
                "”": '"',
                "‘": "'",
                "’": "'",
                "–": "-",
                "—": "-",
                "…": "...",
                "\u00a0": " ",
            }
        )
        self.legal_boilerplate_pattern = re.compile(
            r"(?im)^.*("
            r"copyright|©|all rights reserved|no part of this|permission of the publisher|"
            r"isbn[\s:-]*\S+|digital edition|first published|published in print|published by|"
            r"illustration|illustrations by|trademarks"
            r").*?(\n|$)",
            flags=re.MULTILINE | re.IGNORECASE,
        )
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def _list_files_by_extension(
        self,
        path: str,
        extension: str,
    ) -> Dict[str, List[str]]:
        walks = [
            (os.path.basename(dirpath), dirpath, filenames)
            for dirpath, _, filenames in os.walk(path)
        ]

        return {
            folder: [
                os.path.join(dirpath, file)
                for file in filenames
                if file.endswith(extension)
            ]
            for folder, dirpath, filenames in walks
            if folder != ""
        }

    def _get_book_number(self, filename: str) -> int:
        match = re.search(r"(\d+)(?=\.epub$)", filename)
        if match:
            return int(match.group(1))
        self.logger.warning(f"{filename} has no number.")
        return 9999

    def _extract_epub_books(self, epub_list: List[str]) -> str:
        epub_list_sorted = sorted(epub_list, key=self._get_book_number)
        texts = []
        for epub_path in tqdm(iterable=epub_list_sorted):
            book = epub.read_epub(name=epub_path)
            for idref, _ in book.spine:
                item = book.get_item_with_id(uid=idref)
                if (
                    item is not None
                    and item.get_type() == ebooklib.ITEM_DOCUMENT
                    and item.get_name() not in self.cfg.exclude_files
                ):
                    soup = BeautifulSoup(item.get_body_content(), "html.parser")
                    texts.append(soup.get_text("\n"))

        return "\n".join(texts)

    def _clean_text(self, text: str) -> str:
        cleaned_text = unicodedata.normalize("NFKC", text).translate(self.trans_table)
        cleaned_text = cleaned_text.replace("&nbsp;", " ").replace("&amp;", "&")
        cleaned_text = re.sub(r"\*{2,}", "<scene_break>", cleaned_text)
        cleaned_text = re.sub(r"[-_]{3,}", "", cleaned_text)
        cleaned_text = self.legal_boilerplate_pattern.sub("", cleaned_text)
        cleaned_text = re.sub(
            r"^\s*(page|chapter)\s*([0-9]+|[A-Z]+|[A-Za-z\-]+)\s*$",
            "",
            cleaned_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        cleaned_text = re.sub(
            r"\b(?:https?://|www\.)\S+|\b[a-zA-Z0-9.-]+\.(com|org|net|io|co|sg|gov)(/\S*)?",
            "",
            cleaned_text,
        )
        cleaned_text = re.sub(r"-\n\s*", "", cleaned_text)
        cleaned_text = re.sub(r"([!?.])\1{2,}", r"\1\1\1", cleaned_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\n{2,}", "\n", cleaned_text)
        return cleaned_text.strip()

    def _save_file(self, processed_path: str, title: str, text: str) -> None:
        processed_dir = os.path.join(processed_path, title)
        os.makedirs(name=processed_dir, exist_ok=True)
        processed_path = f"{processed_dir}/{title}.txt"
        with open(file=processed_path, mode="w", encoding="utf-8") as f:
            f.write(text)

    def perform_processing(self):
        epub_dict = self._list_files_by_extension(
            path=self.cfg.raw_epub_dir, extension=".epub"
        )
        for title, epub_list in epub_dict.items():
            self.logger.info(f"Cleaning '{title}' books.")
            raw_text = self._extract_epub_books(epub_list=epub_list)
            clean_text = self._clean_text(text=raw_text)
            self._save_file(
                processed_path=self.cfg.processed_dir,
                title=title,
                text=clean_text,
            )
        txt_dict = self._list_files_by_extension(
            path=self.cfg.processed_dir, extension=".txt"
        )
        print(txt_dict)
