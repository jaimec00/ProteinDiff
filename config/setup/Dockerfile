# syntax=docker/dockerfile:1.6

FROM mambaorg/micromamba:1.5.8 AS mmb

#################################
# Stage 1: builder (CUDA devel) #
#################################
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/conda \
    MAMBA_NO_BANNER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    # speedups for torch cpp extensions
    MAX_JOBS=8 \
    USE_NINJA=1 \
    # keep NVCC via ccache when present
    CUDA_NVCC_EXECUTABLE="ccache nvcc"

# Core build deps only in builder
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates build-essential ninja-build python3-dev python3-pip ccache \
    && rm -rf /var/lib/apt/lists/*

# Micromamba binary
COPY --from=mmb /bin/micromamba /usr/local/bin/micromamba

# Conda env specification
COPY config/setup/environment.yml /tmp/environment.yml

# Create the env (downloads cached across builds)
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n ProteinDiff_env -f /tmp/environment.yml && \
    micromamba clean --all --yes && \
    rm -f /tmp/environment.yml

# Activate-like PATH
ENV PATH=/opt/conda/envs/ProteinDiff_env/bin:$PATH

# Build FlashAttention wheel with robust caching
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.ccache \
    micromamba run -n ProteinDiff_env bash -lc '\
      set -euo pipefail && \
      git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
      cd /tmp/flash-attention && \
      git reset --hard 5059fd5 && \
      python setup.py install && \
      rm -rf /tmp/flash-attention \
    ' && \
    micromamba clean --all --yes

################################
# Stage 2: runtime (CUDA slim) #
################################
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/conda \
    MAMBA_NO_BANNER=1

# Minimal runtime deps only
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

# Micromamba (small) for env activation/ops
COPY --from=mmb /bin/micromamba /usr/local/bin/micromamba

# Copy the fully-prepared env from builder (already contains flash-attn wheel)
COPY --from=builder /opt/conda/envs /opt/conda/envs

# Clean stray caches (usually none after builder clean, but safe)
RUN rm -rf /root/.cache /tmp/*

# add to path
ENV PATH=/opt/conda/envs/ProteinDiff_env/bin:$PATH

# Non-root
RUN useradd -m -u 1000 -s /bin/bash coolguy
USER coolguy
WORKDIR /ProteinDiff

# (optional) tini for PID 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
