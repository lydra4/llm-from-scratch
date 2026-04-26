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

        while len(tokens) >= 2:
            pairs = zip(tokens, tokens[1:])
            best_pair = min(pairs, key=lambda p: vocab.get(p[0] + p[1], float("inf")))
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

            tokens = new_tokens
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
