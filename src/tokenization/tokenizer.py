import ast
import json
import logging
from os import PathLike
from pathlib import Path
from typing import Iterator, Optional

from omegaconf import DictConfig
from tqdm import tqdm


class Tokenizer:
    def __init__(
        self, cfg: DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

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

    def _encode(
        self,
        text: str,
        vocab: dict[bytes, int],
        encoding: str = "utf-8",
    ) -> list[int]:
        text_bytes = text.encode(encoding=encoding)
        tokens = [bytes([b]) for b in text_bytes]

        with tqdm(total=(len(tokens) - 1), desc="Merging tokens", unit="iter") as pbar:
            while len(tokens) >= 2:
                lowest_pair_idx = None
                lowest_vocab_id = float("inf")

                for i in range(len(tokens) - 1):
                    pair = tokens[i] + tokens[i + 1]

                    if pair in vocab and vocab[pair] < lowest_vocab_id:
                        lowest_pair_idx = i
                        lowest_vocab_id = vocab[pair]

                if lowest_pair_idx is None:
                    break

                i = lowest_pair_idx
                merged_token = tokens[i] + tokens[i + 1]
                tokens = tokens[:i] + [merged_token] + tokens[i + 2 :]

                pbar.update(1)

        return [vocab[t] for t in tokens if t in vocab]

    def decode(
        self,
        token_ids: list[int],
        inverse_vocab: dict[int, bytes],
        encoding: str = "utf-8",
    ) -> str:
        byte_sequence = b"".join([inverse_vocab.get(idx, b"") for idx in token_ids])
        return byte_sequence.decode(encoding=encoding, errors="replace")

    def encode_all_text(self):
        vocab = self._parse_vocab_json(vocab_path=self.cfg.vocab_path)

        for split_name, text_content in tqdm(
            self._yield_dataset_paths(data_path=self.cfg.data_path),
            desc="Overall Progress",
        ):
            self.logger.info(f"Encoding '{split_name}' file.")
            tokens_ids = self._encode(text=text_content, vocab=vocab)
            print(tokens_ids)
