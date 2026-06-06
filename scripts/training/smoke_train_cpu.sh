#!/usr/bin/env bash
set -euo pipefail

python src/train.py \
  hardware.device=cpu \
  train.data_path=./data/05-pretokenized/test/test.npy \
  val.data_path=./data/05-pretokenized/val/val.npy \
  dataset.context_window=8 \
  train.loader.batch_size=512 \
  train.loader.shuffle=False \
  val.loader.batch_size=512 \
  model.epochs=1 \
  model.d_model=16 \
  model.n_layers=1 \
  model.n_heads=4 \
  model.dropout=0.0 \
  model.mlp_hidden_dim=64 \
  checkpoint.enabled=true \
  checkpoint.dir=/tmp/llm-smoke-checkpoints \
  checkpoint.checkpoint_interval=1 \
  checkpoint.save_best=true \
  checkpoint.save_latest=true \
  checkpoint.resume_path=null \
  mlflow.tracking_uri=file:/tmp/llm-smoke-mlruns \
  mlflow.experiment_name=llm-smoke \
  mlflow.log_model=false \
  hydra.run.dir=/tmp/llm-smoke-hydra
