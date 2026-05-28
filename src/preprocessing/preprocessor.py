import logging
import os
import re
import unicodedata
from os import PathLike

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from omegaconf import DictConfig
from tqdm import tqdm


class DataPreprocessor:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger | None = None,
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
    ) -> dict[str, list[str]]:
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

    def _validate_epub(self, epub_path: str) -> bool:
        if not epub_path.endswith(".epub"):
            raise ValueError(f"Not an EPUB file: {epub_path}.")

        if not os.path.exists(epub_path):
            raise FileNotFoundError(f"EPUB not found: {epub_path}.")

        try:
            book = epub.read_epub(epub_path)
            if len(book.spine) == 0:
                raise ValueError("EPUB has not content (empty spine).")
            self.logger.info(f"EPUB valid: {len(book.spine)} items in spine.")
            return True
        except Exception as e:
            raise ValueError(f"Invalid EPUB: {e}.")

    def _validate_raw_text(self, text: str, text_label: str = "text") -> None:
        if not isinstance(text, str):
            raise TypeError(f"Expected str for {text_label}, got {type(text)}")

        if len(text.strip()) == 0:
            raise ValueError(f"{text_label} is empty after stripping whitespace")

        if "\ufffd" in text:
            self.logger.warning(
                f"{text_label} contains replacement characters (encoding issues)"
            )

        try:
            text.encode("utf-8").decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"{text_label} contains invalid UTF-8: {e}") from e

        self.logger.info(f"{text_label} validation passed: {len(text)} chars")

    def _validate_split_ratio(
        self, train_ratio: float, val_ratio: float, test_ratio: float
    ) -> None:
        for ratio, name in [
            (train_ratio, "train"),
            (val_ratio, "val"),
            (test_ratio, "test"),
        ]:
            if not isinstance(ratio, (int, float)):
                raise TypeError(f"{name}_ratio must be numeric, got {type(ratio)}.")
            if not 0 <= ratio <= 1:
                raise ValueError(f"{name}_ratio must be in [0,1], got {ratio}.")

            total = train_ratio + val_ratio + test_ratio
            if not abs(total - 1.0) < 1e-9:
                raise ValueError(f"Ratios must sum to 1.0, got {total}.")

            min_ratio = min(train_ratio, val_ratio, test_ratio)
            if min_ratio < 0.1:
                self.logger.warning(
                    f"Imbalanced split detected: min ratio = {min_ratio}."
                )

    def _validate_dataset_splits(
        self,
        train_text: str,
        val_text: str,
        test_text: str,
    ) -> dict[str, int]:
        splits = {"train": train_text, "val": val_text, "test": test_text}
        stats = {}

        for split_name, text in splits.items():
            word_count = len(text.split())

            if word_count == 0:
                raise ValueError(f"{split_name} split is empty!")

            stats[split_name] = word_count
            self.logger.info(f"{split_name}: {word_count} words.")

        total_words = sum(stats.values())
        for split_name, count in stats.items():
            ratio = count / total_words
            self.logger.info(f"{split_name} ratio: {ratio:.1f%}")

        return stats

    def _validate_saved_file(self, file_path: str | PathLike) -> bool:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not saved: {file_path}.")

        try:
            with open(file=file_path, mode="r", encoding="utf-8") as f:
                content = f.read(1_000)
                if len(content) == 0:
                    raise ValueError("File is empty.")
        except UnicodeDecodeError as e:
            raise ValueError(f"File encoding error: {e}.")

        self.logger.info(f"File validation passed: {file_path}.")
        return True

    def _extract_epub_books(self, epub_list: list[str]) -> str:
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

    def _save_processed_text(self, processed_path: str, title: str, text: str) -> None:
        self._validate_raw_text(text=text, text_label=f"processed text ({title})")

        processed_dir = os.path.join(processed_path, title)
        os.makedirs(name=processed_dir, exist_ok=True)
        file_path = os.path.join(processed_dir, f"{title}.txt")

        try:
            with open(file=file_path, mode="w", encoding="utf-8") as f:
                f.write(text)
            self._validate_saved_file(file_path=file_path)
            self.logger.info(f"Successfully saved: {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving {file_path}: {e}")
            raise

    def _load_text_files(self, text_dictionary: dict[str, list[str]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for title, paths in text_dictionary.items():
            try:
                if not paths:
                    raise ValueError(f"No text files found for {title}.")

                texts = []
                for path in sorted(paths):
                    with open(file=path, mode="r", encoding="utf-8") as f:
                        text = f.read()

                    self._validate_raw_text(
                        text=text,
                        text_label=f"loaded file ({title}: {path})",
                    )
                    texts.append(text)

                result[title] = "\n".join(texts)

            except FileNotFoundError as e:
                self.logger.error(f"File not found for {title}: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Error loading {title}: {e}")
                raise

        return result

    def _train_val_test_split(
        self,
        text_map: dict[str, str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple[str, str, str]:
        self._validate_split_ratio(
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )

        self.logger.info(
            f"Performing train/val/test split in ratio:"
            f"{train_ratio:.1f%}/{val_ratio:.1f%}/{test_ratio:.1f%}"
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

        train_str = " ".join(train_text)
        val_str = " ".join(val_text)
        test_str = " ".join(test_text)

        self._validate_dataset_splits(
            train_text=train_str,
            val_text=val_str,
            test_text=test_str,
        )

        self.logger.info("'Train/val/test' split completed.")
        return train_str, val_str, test_str

    def _save_dataset_splits(self, path: str, **kwargs: str) -> None:
        self.logger.info(
            f"Saving '{','.join(key.split(sep='_')[0] for key in kwargs)}' at {path}."
        )
        split_names = [key.split(sep="_")[0] for key in kwargs]
        self.logger.info(f"Saving '{','.join(split_names)}' at {path}")

        for key, value in kwargs.items():
            cleaned_folder_name = key.split(sep="_")[0]
            save_path = os.path.join(path, cleaned_folder_name)
            os.makedirs(name=save_path, exist_ok=True)
            text_path = os.path.join(save_path, cleaned_folder_name + ".txt")

            try:
                self._validate_raw_text(
                    text=value, text_label=f"{cleaned_folder_name} split"
                )
                with open(file=text_path, mode="w", encoding="utf-8") as f:
                    f.write(value)
                self._validate_saved_file(file_path=text_path)
                self.logger.info(f"Suuccessfully saved '{cleaned_folder_name}'")
            except Exception as e:
                self.logger.error(f"Error saving {cleaned_folder_name}: {e}")
                raise

    def preprocess_dataset(self):
        try:
            self.logger.info("Starting dataset preprocessing.")

            epub_dict = self._list_files_by_extension(
                path=self.cfg.raw_epub_dir,
                extension=".epub",
            )

            if not epub_dict:
                raise ValueError(f"No EPUB files found in {self.cfg.raw_epub_dir}")

            for title, epub_list in tqdm(epub_dict.items()):
                self.logger.info(f"Processing '{title}' books ({len(epub_list)} files)")
                raw_text = self._extract_epub_books(epub_list=epub_list)
                clean_text = self._clean_text(text=raw_text)
                self._save_processed_text(
                    processed_path=self.cfg.processed_dir, title=title, text=clean_text
                )

                txt_dict = self._list_files_by_extension(
                    path=self.cfg.processed_dir, extension=".txt"
                )

                if not txt_dict:
                    raise ValueError(
                        f"No processed text files found in {self.cfg.processed_dir}"
                    )

                text_map = self._load_text_files(text_dictionary=txt_dict)
                train_text, val_text, test_text = self._train_val_test_split(
                    text_map=text_map,
                    train_ratio=self.cfg.train_ratio,
                    val_ratio=self.cfg.val_ratio,
                    test_ratio=self.cfg.test_ratio,
                )
                self._save_dataset_splits(
                    path=self.cfg.dataset_dir,
                    train_text=train_text,
                    val_text=val_text,
                    test_text=test_text,
                )

                self.logger.info("Dataset preprocessing completed successfully.")

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise
