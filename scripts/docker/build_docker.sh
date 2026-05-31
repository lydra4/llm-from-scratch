#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-cpu}"

if [[ "$TARGET" != "cpu" && "$TARGET" != "gpu" ]]; then
    echo "Usage: $0 [cpu|gpu]"
    exit 1
fi

docker build \
    -f docker/llmfromscratch.Dockerfile \
    --build-arg TARGET="$TARGET" \
    -t "llmfromscratch:$TARGET" \
    .
