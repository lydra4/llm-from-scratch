import json
import logging
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

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
        byte_to_token_id = dict(zip(byte_content, token_ids))
        token_id_to_bytes = {v: k for k, v in byte_to_token_id.items()}

        return byte_to_token_id, token_id_to_bytes

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
            leave=False,
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
        byte_to_id: Dict[bytes, int],
        id_to_bytes: Dict[int, bytes],
        bigram_freq: Dict[Tuple[int, int], int],
    ) -> Tuple[bytes, bytes, bytes]:
        new_token_id = max(byte_to_id.values()) + 1
        id1, id2 = max(bigram_freq, key=lambda k: bigram_freq[k])

        byte_1 = id_to_bytes[id1]
        byte_2 = id_to_bytes[id2]
        merged_token = byte_1 + byte_2

        byte_to_id[merged_token] = new_token_id
        id_to_bytes[new_token_id] = merged_token

        return byte_1, byte_2, merged_token

    def _replace_pair(
        self,
        byte_content: List[bytes],
        pair: Sequence[bytes],
        new_token: bytes,
        token_ids: List[int],
        byte_to_token_id: Dict[bytes, int],
    ) -> Tuple[List[bytes], List[int]]:
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

        token_ids = [byte_to_token_id[b] for b in byte_content]
        return tokens, token_ids

    def _save_dict(self, dict_to_save: Dict, path: str) -> None:
        os.makedirs(name=path, exist_ok=True)
        with open(file=path, mode="w") as file:
            json.dump(dict_to_save, file, indent=4)

    def tokenize_text(self) -> None:
        self.logger.info("Tokenizing Text")
        byte_content, token_ids = self._convert_text_to_bytes(
            text=self.train_text,
            encoding=self.cfg.character_encoding,
        )
        byte_to_token_id, token_id_to_bytes = self._init_vocab(
            byte_content=byte_content, token_ids=token_ids
        )
        num_merges_to_do = self.cfg.vocab_size - len(byte_to_token_id)

        with tqdm(total=num_merges_to_do, desc="Performing BPE merges") as pbar:
            while len(byte_to_token_id) < self.cfg.vocab_size:
                bigram_freq = self._count_adjacent_token_pairs(token_ids=token_ids)
                *pair, new_token = self._add_top_pair_to_vocab(
                    byte_to_id=byte_to_token_id,
                    id_to_bytes=token_id_to_bytes,
                    bigram_freq=bigram_freq,
                )
                byte_content, token_ids = self._replace_pair(
                    byte_content=byte_content,
                    pair=pair,
                    new_token=new_token,
                    token_ids=token_ids,
                    byte_to_token_id=byte_to_token_id,
                )
                pbar.update(1)

        self._save_dict(dict_to_save=byte_to_token_id, path=self.cfg.dict_save_path)
