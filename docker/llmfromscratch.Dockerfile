FROM python:3.12.10-slim-bookworm AS builder

ARG DEBIAN_FRONTEND="noninteractive"
ARG TARGET=cpu

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY prod-requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r prod-requirements.txt

RUN if [ "$TARGET" = "gpu" ]; then \
    pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124; \
    else \
    pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

FROM python:3.12.10-slim-bookworm

ARG NON_ROOT_USER="template"
ARG NON_ROOT_UID="2222"
ARG NON_ROOT_GID="2222"
ARG HOME_DIR="/home/${NON_ROOT_USER}"
ARG REPO_DIR="."

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${NON_ROOT_GID} ${NON_ROOT_USER} && \
    useradd -l -m -s /bin/bash \
    -u ${NON_ROOT_UID} \
    -g ${NON_ROOT_GID} \
    ${NON_ROOT_USER}

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${HOME_DIR}/.local/bin:${PATH}"
ENV PYTHONIOENCODING=utf8

USER ${NON_ROOT_USER}
WORKDIR ${HOME_DIR}

COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR} .

ENTRYPOINT ["python3"]
