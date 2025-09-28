# syntax=docker/dockerfile:1

# Match CUDA version
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba (tiny bootstrap)
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin/ bin/micromamba --strip-components=2

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    MAMBA_NO_BANNER=1
SHELL ["/bin/bash", "-lc"]

WORKDIR /app
COPY config/setup/environment.yml /app/

RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba create -y -n app -f /app/environment.yml && \
    micromamba clean -a -y

ENV PATH=/opt/conda/envs/app/bin:$PATH

COPY src/* /app
COPY config/train.yml /app
RUN useradd -m appuser && chown -R appuser:appuser /app /opt/conda
USER appuser

# Runtime requires NVIDIA Container Toolkit on host; run with:
# docker run --gpus all yourimage
CMD ["python", "training/train/train.py --config train.yml"]
