import ast
import concurrent.futures
import io
import itertools
import json
import logging
import multiprocessing
import os
import tokenize
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


class BPETokenizer:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def decode(
        self,
        token_ids: list[int],
        inverse_vocab: dict[int, bytes],
        encoding: str = "utf-8",
    ) -> str:
        missing_ids = [idx for idx in token_ids if idx not in inverse_vocab]
        if missing_ids:
            raise KeyError(f"Token ids missing from inverse vocab: {missing_ids[:10]}")

        byte_sequence = b"".join([inverse_vocab[idx] for idx in token_ids])
        return byte_sequence.decode(encoding=encoding)

    def _parse_merge_pairs(self, raw_pair: str) -> tuple[str, str]:
        string_tokens = []

        for token in tokenize.generate_tokens(io.StringIO(raw_pair).readline):
            if token.type == tokenize.STRING:
                string_tokens.append(token.string)

        if len(string_tokens) != 2:
            raise ValueError(
                f"Expected merge pair to contain 2 bytes literals, got {len(string_tokens)}: {raw_pair!r}"
            )

        return string_tokens[0], string_tokens[1]

    def _parse_merges_json(
        self,
        merges_path: str | PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
    ) -> dict[tuple[bytes, bytes], int]:
        with open(file=merges_path, mode=mode, encoding=encoding) as f:
            raw_merges = json.load(f)

        merges = {}
        for raw_pair, rank in raw_merges.items():
            left, right = self._parse_merge_pairs(raw_pair=raw_pair)
            merges[(ast.literal_eval(left), ast.literal_eval(right))] = int(rank)

        return merges

    def _validate_merges(
        self,
        merges: dict[tuple[bytes, bytes], int],
        vocab: dict[bytes, int],
    ) -> None:
        ranks = list(merges.values())

        if len(ranks) != len(set(ranks)):
            raise ValueError("Duplicate BPE merge ranks found.")

        expected_ranks = set(range(len(ranks)))
        actual_ranks = set(ranks)
        if actual_ranks != expected_ranks:
            raise ValueError(
                f"BPE merge ranks must be contiguous from 0 to {len(ranks) - 1}."
            )

        for left, right in merges:
            merged = left + right
            if merged not in vocab:
                raise ValueError(f"Merged token {merged!r} is missing from vocab.")

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

        vocab = {}
        for raw_token, raw_token_id in raw_vocab.items():
            token = ast.literal_eval(raw_token)

            if not isinstance(token, bytes):
                raise TypeError(
                    f"Expected vobab key to decode to bytes, got {type(token)}"
                )

            vocab[token] = int(raw_token_id)

        return vocab

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

    def _save_token_ids_as_array(
        self,
        tokens_ids: list[int],
        path: str | PathLike,
        filename: str,
    ) -> None:
        self.logger.info(f"Preparing to save '{filename}' tokens")

        dtype = np.dtype(self.cfg.get("token_dtype", "int32"))
        tokens_array = np.array(object=tokens_ids, dtype=dtype)

        save_path = os.path.join(path, filename)
        os.makedirs(name=save_path, exist_ok=True)

        full_filepath = os.path.join(save_path, f"{filename}.npy")
        np.save(file=full_filepath, arr=tokens_array)

        self.logger.info(f"Saving of '{filename}' successfull.")

    @staticmethod
    def _encode(
        text: str,
        vocab: dict[bytes, int],
        merges: dict[tuple[bytes, bytes], int],
        encoding: str = "utf-8",
        disable_pbar: bool = False,
        pbar_desc: str = "BPE Merges",
    ) -> list[int]:
        tokens = [bytes([b]) for b in text.encode(encoding=encoding)]

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
                pairs = itertools.pairwise(tokens)

                best_pair = min(
                    pairs,
                    key=lambda pair: merges.get(pair, float("inf")),
                )

                if best_pair not in merges:
                    break

                merged_best = best_pair[0] + best_pair[1]
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

        missing_tokens = [token for token in tokens if token not in vocab]
        if missing_tokens:
            raise KeyError(f"Encoded tokens missing from vocab: {missing_tokens[:10]}")

        return [vocab[token] for token in tokens]

    def _encode_parallel(
        self,
        vocab: dict[bytes, int],
        merges: dict[tuple[bytes, bytes], int],
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
                    merges=merges,
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
                    self._save_token_ids_as_array(
                        tokens_ids=token_ids,
                        path=path,
                        filename=split_name,
                    )
                except Exception as e:
                    self.logger.error(f"Error encoding {split_name}:{e}.")
                    raise RuntimeError(f"Failed to encode split: {split_name}") from e

    def _encode_sequential(
        self,
        vocab: dict[bytes, int],
        merges: dict[tuple[bytes, bytes], int],
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
                merges=merges,
                disable_pbar=False,
                pbar_desc=f"Merges | {split_name}",
            )
            self._save_token_ids_as_array(
                tokens_ids=token_ids,
                path=path,
                filename=split_name,
            )

    def encode_all_text(self) -> None:
        vocab = self._parse_vocab_json(vocab_path=self.cfg.vocab_path)
        merges = self._parse_merges_json(merges_path=self.cfg.merges_path)
        self._validate_merges(merges=merges, vocab=vocab)

        datasets = list(self._yield_dataset_paths(data_path=self.cfg.data_path))

        if self.cfg.use_multiprocessing:
            self._encode_parallel(
                vocab=vocab,
                merges=merges,
                datasets=datasets,
                path=self.cfg.save_path,
            )

        else:
            self._encode_sequential(
                vocab=vocab,
                merges=merges,
                datasets=datasets,
                path=self.cfg.save_path,
            )
