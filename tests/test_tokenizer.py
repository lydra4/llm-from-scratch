import json
import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tokenizer.bpe_builder import BPEBuilder
from tokenizer.bpe_tokenizer import BPETokenizer


def test_bpe_builder_init_byte_vocab(logger: logging.Logger) -> None:
    builder = BPEBuilder(cfg=OmegaConf.create({}), logger=logger)

    byte_to_id, id_to_bytes = builder._init_vocab()

    assert len(byte_to_id) == 256
    assert len(id_to_bytes) == 256
    assert byte_to_id[b"a"] == 97
    assert id_to_bytes[97] == b"a"


def test_bpe_encode_applies_best_merge() -> None:
    vocab = {bytes([i]): i for i in range(256)}
    vocab[b"ab"] = 256
    merges = {(b"a", b"b"): 0}

    token_ids = BPETokenizer._encode(
        text="ab",
        vocab=vocab,
        merges=merges,
        disable_pbar=True,
    )

    assert token_ids == [256]


def test_bpe_decode_reconstructs_text(logger: logging.Logger) -> None:
    tokenizer = BPETokenizer(cfg=OmegaConf.create({}), logger=logger)

    decoded = tokenizer.decode(
        token_ids=[256, 99],
        inverse_vocab={256: b"ab", 99: b"c"},
    )

    assert decoded == "abc"


def test_bpe_decode_raises_for_missing_token_id(logger: logging.Logger) -> None:
    tokenizer = BPETokenizer(cfg=OmegaConf.create({}), logger=logger)

    with pytest.raises(KeyError, match="Token ids missing"):
        tokenizer.decode(token_ids=[999], inverse_vocab={})


def test_parse_vocab_json_loads_bytes_keys(
    tmp_path: Path,
    logger: logging.Logger,
) -> None:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"b'a'": 97, "b'ab'": 256}), encoding="utf-8")

    tokenizer = BPETokenizer(cfg=OmegaConf.create({}), logger=logger)
    vocab = tokenizer._parse_vocab_json(vocab_path=vocab_path)

    assert vocab == {b"a": 97, b"ab": 256}


def test_parse_merges_json_loads_byte_pairs(
    tmp_path: Path,
    logger: logging.Logger,
) -> None:
    merges_path = tmp_path / "merges.json"
    merges_path.write_text(json.dumps({"b'a' b'b'": 0}), encoding="utf-8")

    tokenizer = BPETokenizer(cfg=OmegaConf.create({}), logger=logger)

    merges = tokenizer._parse_merges_json(merges_path=merges_path)

    assert merges == {(b"a", b"b"): 0}
