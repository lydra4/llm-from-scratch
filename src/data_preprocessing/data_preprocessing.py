import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

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

    def _load_text_files(self, text_dictionary: Dict[str, List[str]]) -> Dict[str, str]:
        return {
            title: open(file=paths[0], mode="r", encoding="utf-8").read()
            for title, paths in text_dictionary.items()
        }

    def _train_val_test_split(
        self,
        text_map: Dict[str, str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Tuple[str, str, str]:
        total = train_ratio + val_ratio + test_ratio
        if not abs(total - 1.0) < 1e-9:
            raise ValueError(f"Train/val/test ratios must sum to 1. Got: '{total}'.")

        self.logger.info(
            f"Performing train/val/test split in the ratio: '{train_ratio}/{val_ratio}/{test_ratio}'."
        )

        train_text, val_text, test_text = [], [], []
        for value in text_map.values():
            words = value.split()
            n_words = len(words)

            train_end = int(n_words * train_ratio)
            val_end = int(n_words * (train_ratio + val_ratio))

            train_words = words[:train_end]
            val_words = words[train_end:val_end]
            test_words = words[val_end:]

            train_text.extend(train_words)
            val_text.extend(val_words)
            test_text.extend(test_words)

        self.logger.info("'Train/val/test' split completed.")
        return (
            " ".join(train_text),
            " ".join(val_text),
            " ".join(test_text),
        )

    def _save_text_files(self, path: str, **kwargs: str) -> None:
        self.logger.info(
            f"Saving '{','.join(key.split(sep='_')[0] for key in kwargs)}' at {path}."
        )
        for key, value in kwargs.items():
            cleaned_folder_name = key.split(sep="_")[0]
            save_path = os.path.join(path, cleaned_folder_name)
            os.makedirs(name=save_path, exist_ok=True)
            text_path = os.path.join(save_path, cleaned_folder_name + ".txt")
            with open(file=text_path, mode="w", encoding="utf-8") as f:
                f.write(value)
            self.logger.info(f"Successfully saved '{cleaned_folder_name}'.")

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
            path=self.cfg.processed_dir,
            extension=".txt",
        )
        text_map = self._load_text_files(text_dictionary=txt_dict)
        train_text, val_text, test_text = self._train_val_test_split(
            text_map=text_map,
            train_ratio=self.cfg.train_ratio,
            val_ratio=self.cfg.val_ratio,
            test_ratio=self.cfg.test_ratio,
        )
        self._save_text_files(
            path=self.cfg.dataset_dir,
            train_text=train_text,
            val_text=val_text,
            test_text=test_text,
        )
