# syntax=docker/dockerfile:1.6

# setup nvidia env with micromamba
FROM mambaorg/micromamba:1.5.8 AS mmb
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS runtime

# Ensure noninteractive apt
ENV DEBIAN_FRONTEND=noninteractive

# --- Install OS deps needed during build/runtime ---
# Use BuildKit caches for apt metadata and package cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# copy micromamba bin to nvidia env
COPY --from=mmb /bin/micromamba /usr/local/bin/micromamba
COPY config/setup/environment.yml /tmp/

# where mamba stuff goes and no banner
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    MAMBA_NO_BANNER=1

# create env, then delete the env file and cache
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n ProteinDiff_env -f /tmp/environment.yml && \
    micromamba clean --all --yes && \
    rm -f /tmp/environment.yml


# Build & install FA3
RUN --mount=type=cache,target=/root/.cache/pip \
    micromamba run -n ProteinDiff_env bash -lc '\
        set -euo pipefail && \
        git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
        cd /tmp/flash-attention && \
        git reset --hard 5059fd5 && \
        cd hopper && \
        python setup.py install \
    ' && \
    rm -rf /tmp/flash-attention


# edit paths
ENV PATH=/opt/conda/envs/ProteinDiff_env/bin:$PATH

# no root user
RUN useradd -m -u 1000 -s /bin/bash coolguy
USER coolguy

WORKDIR /ProteinDiff/