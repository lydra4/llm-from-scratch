import collections
import json
import logging
import os
from collections.abc import Sequence

from omegaconf import DictConfig
from tqdm import tqdm


class BPEBuilder:
    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def _save_json(self, dict_to_save: dict, path: str | os.PathLike) -> None:
        folder = os.path.dirname(path)
        os.makedirs(name=folder, exist_ok=True)

        with open(file=path, mode="w", encoding="utf-8") as file:
            json.dump(
                dict_to_save,
                file,
                indent=4,
                ensure_ascii=False,
            )

    def _load_training_text(
        self,
        data_path: str,
        encoding: str = "utf-8",
        mode: str = "r",
    ) -> str:
        train_path = os.path.join(data_path, "train", "train.txt")
        with open(
            file=train_path,
            encoding=encoding,
            mode=mode,
        ) as f:
            train_text = f.read()

        return train_text

    def _convert_text_to_bytes(
        self,
        text: str,
        encoding: str = "utf-8",
    ) -> tuple[list[bytes], list[int]]:
        byte_sequence = text.encode(encoding)
        byte_content = [bytes([byte]) for byte in byte_sequence]

        token_ids = list(byte_sequence)
        return byte_content, token_ids

    def _init_vocab(self) -> tuple[dict[bytes, int], dict[int, bytes]]:
        byte_to_token_id = {bytes([i]): i for i in range(256)}
        token_id_to_bytes = {i: bytes([i]) for i in range(256)}

        return byte_to_token_id, token_id_to_bytes

    def _count_bigram_frequencies(
        self,
        token_ids: list[int],
    ) -> dict[
        tuple[int, int],
        int,
    ]:
        bigram_freq = collections.Counter(
            tqdm(
                zip(token_ids, token_ids[1:]),
                total=len(token_ids) - 1,
            )
        )
        sorted_bigram_freq = dict(bigram_freq.most_common())

        return sorted_bigram_freq

    def _add_top_pair_to_vocab(
        self,
        byte_to_id: dict[bytes, int],
        id_to_bytes: dict[int, bytes],
        bigram_freq: dict[tuple[int, int], int],
    ) -> tuple[bytes, bytes, bytes]:
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
        byte_content: list[bytes],
        pair: Sequence[bytes],
        new_token: bytes,
        byte_to_token_id: dict[bytes, int],
    ) -> tuple[list[bytes], list[int]]:
        assert len(pair) == 2, f"Must have only 2 items in pair, got {len(pair)}."

        new_byte_content = []
        i = 0

        while i < len(byte_content):
            if (
                (i < (len(byte_content) - 1))
                and (byte_content[i] == pair[0])
                and (byte_content[i + 1] == pair[1])
            ):
                new_byte_content.append(new_token)
                i += 2
            else:
                new_byte_content.append(byte_content[i])
                i += 1

        new_token_ids = [byte_to_token_id[b] for b in new_byte_content]
        return new_byte_content, new_token_ids

    def _save_vocabulary(self, dict_to_save: dict, path: str) -> None:
        folder = os.path.dirname(path)
        os.makedirs(name=folder, exist_ok=True)

        json_ready_dict = {str(k): v for k, v in dict_to_save.items()}

        with open(file=path, mode="w", encoding="utf-8") as file:
            json.dump(json_ready_dict, file, indent=4, ensure_ascii=False)

    def build_vocabulary(self) -> None:
        self.logger.info("Building BPE Vocabulary")

        train_text = self._load_training_text(data_path=self.cfg.data_path)
        byte_content, token_ids = self._convert_text_to_bytes(text=train_text)

        byte_to_token_id, token_id_to_bytes = self._init_vocab()

        merges: dict[str, int] = {}
        num_merges_to_do = self.cfg.vocab_size - len(byte_to_token_id)

        with tqdm(total=num_merges_to_do, desc="Performing BPE merges") as pbar:
            while len(byte_to_token_id) < self.cfg.vocab_size:
                bigram_freq = self._count_bigram_frequencies(token_ids=token_ids)

                if not bigram_freq:
                    break

                byte_1, byte_2, merged_token = self._add_top_pair_to_vocab(
                    byte_to_id=byte_to_token_id,
                    id_to_bytes=token_id_to_bytes,
                    bigram_freq=bigram_freq,
                )

                merges[f"{byte_1!r} {byte_2!r}"] = len(merges)

                byte_content, token_ids = self._replace_pair(
                    byte_content=byte_content,
                    pair=(byte_1, byte_2),
                    new_token=merged_token,
                    byte_to_token_id=byte_to_token_id,
                )
                pbar.update(1)

        vocab_json = {
            repr(token_bytes): token_id
            for token_bytes, token_id in byte_to_token_id.items()
        }

        self._save_json(dict_to_save=vocab_json, path=self.cfg.dict_save_path)
        self._save_json(dict_to_save=merges, path=self.cfg.merges_save_path)
