import logging

import torch
from omegaconf import DictConfig

from model.transformer import TransformerLM


def test_transformer_forward_returns_logits_shape(
    tiny_cfg: DictConfig,
    logger: logging.Logger,
) -> None:
    model = TransformerLM(cfg=tiny_cfg, logger=logger)
    idx = torch.randint(
        low=0,
        high=tiny_cfg.model.vocab_size,
        size=(2, tiny_cfg.dataset.context_window),
    )

    logits = model(idx)

    assert logits.shape == torch.Size(
        [
            2,
            tiny_cfg.dataset.context_window,
            tiny_cfg.model.vocab_size,
        ]
    )
    assert torch.isfinite(logits).all()


def test_transformer_ties_embedding_and_lm_head_weights(
    tiny_cfg: DictConfig,
    logger: logging.Logger,
) -> None:
    tiny_cfg.model.tie_weights = True

    model = TransformerLM(cfg=tiny_cfg, logger=logger)

    assert (
        model.lm_head.weight.data_ptr()
        == model.embeddings.token_embeddings.weight.data_ptr()
    )


def test_transformer_num_parameters_returns_positive_count(
    tiny_cfg: DictConfig,
    logger: logging.Logger,
) -> None:
    model = TransformerLM(cfg=tiny_cfg, logger=logger)

    assert model.num_parameters(non_embedding=False) > 0
    assert model.num_parameters(non_embedding=True) > 0
