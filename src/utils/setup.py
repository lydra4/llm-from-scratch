import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer

from model.transformer import TransformerLM


def setup_model_and_components(
    cfg: DictConfig,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[
    TransformerLM,
    Optimizer,
    torch.nn.Module,
]:
    model = TransformerLM(cfg=cfg, logger=logger).to(device=device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    criterion = instantiate(cfg.criterion)

    return model, optimizer, criterion
