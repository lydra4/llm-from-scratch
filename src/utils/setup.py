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


def resolve_device(requested_device: str, logger: logging.Logger) -> torch.device:
    requested_device = requested_device.lower()

    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(
                "Resolved device: cuda (%s)",
                torch.cuda.get_device_name(0),
            )
            return device

        device = torch.device("cpu")
        logger.info("Resolved device: cpu")
        return device

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested in config, but no CUDA-capable GPU is available. "
                "Set hardware.device=cpu or hardware.device=auto, or run on a machine with a CUDA GPU."
            )

        device = torch.device("cuda")
        logger.info("Resolved device: cuda (%s)", torch.cuda.get_device_name(0))
        return device

    if requested_device == "cpu":
        device = torch.device("cpu")
        logger.info("Resolved device: cpu")
        return device

    raise ValueError(
        f"Unsupported hardware.device={requested_device!r}. "
        "Expected one of: auto, cuda, cpu."
    )
