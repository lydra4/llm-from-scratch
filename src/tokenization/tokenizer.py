import ast
import json
import logging
from typing import Optional

from omegaconf import DictConfig


class Tokenizer:
    def __init__(
        self, cfg: DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        with open(
            file=self.cfg.vocab_path,
            mode="r",
            encoding=self.cfg.character_encoding,
        ) as f:
            raw_vocab = json.load(f)

        self.vocab = {ast.literal_eval(k): int(v) for k, v in raw_vocab.items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(
        self,
        text: str,
        vocab: dict[bytes, int],
        encoding: str = "utf-8",
    ) -> list[int]:
        text_bytes = text.encode(encoding=encoding)
        tokens = [bytes([b]) for b in text_bytes]

        while len(tokens) >= 2:
            lowest_pair_idx = None
            lowest_vocab_id = float("inf")

            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i + 1]

                if pair in vocab and vocab[pair] < lowest_vocab_id:
                    lowest_vocab_id = vocab[pair]
                    lowest_vocab_id = i

            if lowest_pair_idx is None:
                break

            i = lowest_pair_idx
            merged_token = tokens[i] + tokens[i + 1]
            tokens = tokens[:i] + [merged_token] + tokens[i + 2 :]

        return [vocab[t] for t in tokens if t in vocab]

    def decode(
        self,
        token_ids: list[int],
        inverse_vocab: dict[int, bytes],
        encoding: str = "utf-8",
    ) -> str:
        byte_sequence = b"".join([inverse_vocab.get(idx, b"") for idx in token_ids])
        return byte_sequence.decode(encoding=encoding, errors="replace")
