import ast
import concurrent.futures
import json
import logging
import multiprocessing
import os
from os import PathLike
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


class BPETokenizer:
    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def decode(
        self,
        token_ids: list[int],
        inverse_vocab: dict[int, bytes],
        encoding: str = "utf-8",
    ) -> str:
        byte_sequence = b"".join([inverse_vocab.get(idx, b"") for idx in token_ids])
        return byte_sequence.decode(encoding=encoding, errors="replace")

    def _parse_vocab_json(
        self,
        vocab_path: str | PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
    ) -> dict[bytes, int]:
        with open(
            file=vocab_path,
            mode=mode,
            encoding=encoding,
        ) as f:
            raw_vocab = json.load(f)

        return {ast.literal_eval(k): int(v) for k, v in raw_vocab.items()}

    def _yield_dataset_paths(
        self,
        data_path: str | PathLike,
        extension: str = ".txt",
        encoding: str = "utf-8",
    ) -> Iterator[tuple[str, str]]:
        paths = Path(data_path).rglob(f"*{extension}")

        for path in paths:
            folder_name = path.parent.name
            text_content = path.read_text(encoding=encoding)

            yield folder_name, text_content

    def _save_tokens_list(
        self,
        tokens_ids: list[int],
        path: str | PathLike,
        filename: str,
    ) -> None:
        self.logger.info(f"Preparing to save '{filename}' tokens")

        tokens_array = np.array(object=tokens_ids, dtype=np.int16)
        save_path = os.path.join(path, filename)
        os.makedirs(name=save_path, exist_ok=True)
        full_filepath = os.path.join(save_path, f"{filename}.npy")
        np.save(file=full_filepath, arr=tokens_array)

        self.logger.info(f"Saving of '{filename}' successfull.")

    @staticmethod
    def _encode(
        text: str,
        vocab: dict[bytes, int],
        encoding: str = "utf-8",
        disable_pbar: bool = False,
        pbar_desc: str = "BPE Merges",
    ) -> list[int]:
        text_bytes = text.encode(encoding=encoding)
        tokens = [bytes([b]) for b in text_bytes]

        identity = multiprocessing.current_process()._identity
        pos = identity[0] if identity else 1

        with tqdm(
            total=max(0, len(tokens) - 1),
            desc=pbar_desc,
            leave=False,
            disable=disable_pbar,
            position=pos,
        ) as pbar:
            while len(tokens) >= 2:
                pairs = zip(tokens, tokens[1:])
                best_pair = min(
                    pairs,
                    key=lambda p: vocab.get(p[0] + p[1], float("inf")),
                )
                merged_best = best_pair[0] + best_pair[1]

                if merged_best not in vocab:
                    break

                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (
                        (i < len(tokens) - 1)
                        and (tokens[i] == best_pair[0])
                        and (tokens[i + 1] == best_pair[1])
                    ):
                        new_tokens.append(merged_best)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1

                tokens_removed = len(tokens) - len(new_tokens)
                pbar.update(tokens_removed)
                tokens = new_tokens

        return [vocab[t] for t in tokens if t in vocab]

    def _encode_parallel(
        self,
        vocab: dict[bytes, int],
        datasets: list[tuple[str, str]],
        path: str,
    ) -> None:
        self.logger.info("Encoding using multiprocessing.")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._encode,
                    text=text_content,
                    vocab=vocab,
                    disable_pbar=False,
                    pbar_desc=f"Worker | {split_name}",
                ): split_name
                for split_name, text_content in datasets
            }

            for future in tqdm(
                iterable=concurrent.futures.as_completed(futures),
                total=len(datasets),
                desc="Encoding(Multiprocessing)",
                position=0,
                leave=True,
            ):
                split_name = futures[future]
                try:
                    token_ids = future.result()
                    self._save_tokens_list(
                        tokens_ids=token_ids,
                        path=path,
                        filename=split_name,
                    )
                except Exception as e:
                    self.logger.error(f"Error encoding {split_name}:{e}.")

    def _encode_sequential(
        self,
        vocab: dict[bytes, int],
        datasets: list[tuple[str, str]],
        path: str,
    ) -> None:
        self.logger.info("Encoding using a processor.")

        for split_name, text_content in tqdm(
            iterable=datasets,
            desc="Encoding(Single Processor)",
            position=0,
            leave=True,
        ):
            self.logger.info(f"Encoding '{split_name}' set.")
            token_ids = self._encode(
                text=text_content,
                vocab=vocab,
                disable_pbar=False,
                pbar_desc=f"Merges | {split_name}",
            )
            self._save_tokens_list(
                tokens_ids=token_ids,
                path=path,
                filename=split_name,
            )

    def encode_all_text(self) -> None:
        vocab = self._parse_vocab_json(vocab_path=self.cfg.vocab_path)
        datasets = list(self._yield_dataset_paths(data_path=self.cfg.data_path))

        if self.cfg.use_multiprocessing:
            self._encode_parallel(
                vocab=vocab,
                datasets=datasets,
                path=self.cfg.save_path,
            )

        else:
            self._encode_sequential(
                vocab=vocab,
                datasets=datasets,
                path=self.cfg.data_path,
            )
