from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, cast

import mlflow
import mlflow.pytorch as mlflow_pytorch
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module


def flatten_dict(d: Mapping[str, Any], parent_key: str = "", sep: str = ".") -> dict:
    items: dict[str, Any] = {}

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.update(flatten_dict(d=value, parent_key=new_key, sep=sep))
        else:
            items[new_key] = value

    return items


@contextmanager
def start_mlflow_run(cfg: DictConfig):
    mlflow.set_tracking_uri(uri=cfg.mlflow.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_name)

    resolved_cfg = cast(
        Mapping[str, Any],
        OmegaConf.to_container(cfg=cfg, resolve=True),
    )
    mlflow_params = flatten_dict(d=resolved_cfg)

    with mlflow.start_run():
        mlflow.log_params(mlflow_params)
        yield


def log_epoch_metrics(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    epoch: int,
) -> None:
    metrics = {
        "train_loss": train_metrics["loss"],
        "train_samples_per_sec": train_metrics["samples_per_sec"],
        "train_tokens_per_sec": train_metrics["tokens_per_sec"],
        "train_epoch_time_sec": train_metrics["epoch_time_sec"],
        "train_num_batches": train_metrics["num_batches"],
        "val_loss": val_metrics["loss"],
        "val_num_batches": val_metrics["num_batches"],
    }

    if "peak_memory_mb" in train_metrics:
        metrics["train_peak_memory_mb"] = train_metrics["peak_memory_mb"]

    mlflow.log_metrics(metrics=metrics, step=epoch)


def log_model(cfg: DictConfig, model: Module, epoch: int) -> None:
    should_log_model = (
        cfg.mlflow.log_model
        and cfg.mlflow.checkpoint_interval > 0
        and (epoch + 1) % cfg.mlflow.checkpoint_interval == 0
    )

    if should_log_model:
        mlflow_pytorch.log_model(model, name="model")
