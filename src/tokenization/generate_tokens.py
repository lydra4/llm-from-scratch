import logging
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from omegaconf import DictConfig
from tqdm import tqdm


class GenerateTokens:
    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        train_path = os.path.join(self.cfg.data_path, "train", "train.txt")
        with open(file=train_path, mode="r", encoding="utf-8") as f:
            self.train_text = f.read()

    def _convert_text_to_bytes(
        self,
        text: str,
        encoding: str,
    ) -> Tuple[List[bytes], List[int]]:
        byte_sequence = text.encode(encoding)
        byte_content = [bytes([byte]) for byte in byte_sequence]

        token_ids = list(byte_sequence)
        return byte_content, token_ids

    def _init_vocab(
        self,
        byte_content: List[bytes],
        token_ids: List[int],
    ) -> Dict[bytes, int]:
        unique_byte_content = list(set(byte_content))
        unique_token_ids = list(set(token_ids))
        token_to_id = {
            byte_content: token_id
            for byte_content, token_id in zip(unique_byte_content, unique_token_ids)
        }
        return token_to_id

    def _count_adjacent_token_pairs(
        self,
        token_ids: List[int],
    ) -> Dict[
        Tuple[int, int],
        int,
    ]:
        bigram_freq: DefaultDict[Tuple[int, int], int] = defaultdict(int)
        for i in tqdm(
            iterable=range(len(token_ids) - 1),
            desc="Calculating bigram frequency",
        ):
            pair = (token_ids[i], token_ids[i + 1])
            bigram_freq[pair] += 1

        return bigram_freq

    def tokenize_text(self) -> Any:
        self.logger.info("Tokenizing Text")
        byte_content, token_ids = self._convert_text_to_bytes(
            text=self.train_text,
            encoding=self.cfg.character_encoding,
        )
        # token_to_id = self._init_vocab(byte_content=byte_content, token_ids=token_ids)
        bigram_freq = self._count_adjacent_token_pairs(token_ids=token_ids)
        print(bigram_freq)
