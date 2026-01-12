import logging
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

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
    ) -> Tuple[Dict[bytes, int], Dict[int, bytes]]:
        vocab = dict(zip(byte_content, token_ids))
        id_to_bytes = {v: k for k, v in vocab.items()}

        return vocab, id_to_bytes

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

        sorted_bigram_freq = dict(
            sorted(
                bigram_freq.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        return sorted_bigram_freq

    def _add_top_pair_to_vocab(
        self,
        vocab: Dict[bytes, int],
        id_to_bytes: Dict[int, bytes],
        bigram_freq: Dict[Tuple[int, int], int],
    ) -> Tuple[bytes, bytes, bytes]:
        new_token_id = max(vocab.values()) + 1
        id1, id2 = max(bigram_freq, key=lambda k: bigram_freq[k])

        byte_1 = id_to_bytes[id1]
        byte_2 = id_to_bytes[id2]
        merged_token = byte_1 + byte_2

        vocab[merged_token] = new_token_id
        id_to_bytes[new_token_id] = merged_token

        return byte_1, byte_2, merged_token

    def _replace_pair(
        self,
        byte_content: List[bytes],
        pair: Sequence[bytes],
        new_token: bytes,
    ) -> List[bytes]:
        assert len(pair) == 2, f"Must have only 2 items in pair, got {len(pair)}."

        tokens = []
        i = 0

        while i < len(byte_content):
            if (
                (i < (len(byte_content) - 1))
                and (byte_content[i] == pair[0])
                and (byte_content[i + 1] == pair[1])
            ):
                tokens.append(new_token)
                i += 2
            else:
                tokens.append(byte_content[i])
                i += 1
        return tokens

    def tokenize_text(self) -> Any:
        self.logger.info("Tokenizing Text")
        byte_content, token_ids = self._convert_text_to_bytes(
            text=self.train_text,
            encoding=self.cfg.character_encoding,
        )
        vocab, id_to_bytes = self._init_vocab(
            byte_content=byte_content, token_ids=token_ids
        )
        bigram_freq = self._count_adjacent_token_pairs(token_ids=token_ids)
        *pair, new_token = self._add_top_pair_to_vocab(
            vocab=vocab,
            id_to_bytes=id_to_bytes,
            bigram_freq=bigram_freq,
        )
        tokens = self._replace_pair(
            byte_content=byte_content,
            pair=pair,
            new_token=new_token,
        )
        print(tokens)
