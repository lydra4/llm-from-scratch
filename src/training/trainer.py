import logging
import math
import time

import torch
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    device: torch.device,
    cfg: DictConfig,
    logger: logging.Logger,
    epoch: int,
) -> dict[str, float]:
    if cfg.model.vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {cfg.model.vocab_size}")

    model.train()
    total_train_loss = 0
    num_batches = 0
    total_tokens = 0
    total_samples = 0

    epoch_start = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
            try:
                if x.size(0) == 0:
                    logger.warning(f"Empty batch at index {batch_idx}, skipping")
                    continue

                x, y = x.to(device), y.to(device)
                total_samples += x.size(0)
                total_tokens += x.numel()

                logits = model(x)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise RuntimeError(f"NaN/Inf in model output at batch {batch_idx}")

                loss = criterion(logits.view(-1, cfg.model.vocab_size), y.view(-1))

                if math.isnan(loss.item()):
                    raise RuntimeError(
                        f"NaN loss at batch {batch_idx}"
                        "Check learning rate, batch data or model"
                    )

                if math.isinf(loss.item()):
                    raise RuntimeError(f"Inf loss at batch {batch_idx}")

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            raise RuntimeError(f"NaN gradient in {name}")
                        if torch.isinf(param.grad).any():
                            raise RuntimeError(f"Inf gradient in {name}")

                optimizer.step()
                total_train_loss += loss.item()
                num_batches += 1

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"OOM at batch {batch_idx}. Reduce batch size.")
                raise RuntimeError("Out of GPU memory") from e
            except RuntimeError as e:
                logger.error(f"Training error at batch {batch_idx}: {e}")
                raise

        epoch_time = time.time() - epoch_start
        avg_train_loss = total_train_loss / max(num_batches, 1)
        samples_per_sec = total_samples / epoch_time if epoch_time > 0 else 0
        tokens_per_sec = total_tokens / epoch_time if epoch_time > 0 else 0

        metrics = {
            "loss": avg_train_loss,
            "samples_per_sec": samples_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "epoch_time_sec": epoch_time,
            "num_batches": num_batches,
        }

        if device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
            metrics["peak_memory_mb"] = peak_memory_mb
            logger.info(f"Peak GPU Memory: {peak_memory_mb:.2f} MB")

        logger.info(
            f"Epoch {epoch} | Loss: {avg_train_loss:.2f} | "
            f"Speed: {tokens_per_sec:.0f} tokens/sec | "
            f"Time: {epoch_time:.2f}s"
        )

        return metrics

    except Exception as e:
        logger.error(f"Fatal error during training epoch {epoch}: {e}")
        raise


def validate_epoch(
    model: Module,
    val_loader: DataLoader,
    criterion: Module,
    device: torch.device,
    cfg: DictConfig,
    logger: logging.Logger,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    total_val_loss = 0
    num_batches = 0

    try:
        logger.info(f"Scoring on validation for epoch: {epoch:.4f}")
        with torch.no_grad():
            for batch_idx, (x_val, y_val) in enumerate(val_loader):
                try:
                    x_val, y_val = x_val.to(device), y_val.to(device)

                    if x_val.size(0) == 0:
                        logger.warning(f"Empty validation batch {batch_idx}")
                        continue

                    val_logits = model(x_val)

                    if torch.isnan(val_logits).any():
                        logger.warning(f"NaN in validation output at batch {batch_idx}")
                        continue

                    val_loss = criterion(
                        val_logits.view(-1, cfg.model.vocab_size), y_val.view(-1)
                    )

                    if math.isnan(val_loss.item()):
                        logger.warning("NaN validation loss at batch, skipping")
                        continue

                    if math.isinf(val_loss.item()):
                        logger.warning("NaN validation loss at batch, skipping")
                        continue

                    total_val_loss += val_loss.item()
                    num_batches += 1

                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue

            if num_batches == 0:
                logger.warning("No validation batches, returning inf")
                avg_val_loss = float("inf")
            else:
                avg_val_loss = total_val_loss / num_batches

            if math.isnan(avg_val_loss):
                logger.warning("Average validation loss is NaN, returning inf")
                avg_val_loss = float("inf")

            logger.info(f"Epoch {epoch} | Val loss: {avg_val_loss:.4f}")

            return {"loss": avg_val_loss, "num_batches": num_batches}

    except Exception as e:
        logger.error(f"Fatal error during validation epoch {epoch}: {e}")
        raise
