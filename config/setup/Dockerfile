# syntax=docker/dockerfile:1.6

#################################
# Stage 1: builder (CUDA devel) #
#################################
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    # speedups for torch cpp extensions
    MAX_JOBS=8 \
    USE_NINJA=1 \
    # keep NVCC via ccache when present
    CUDA_NVCC_EXECUTABLE="ccache nvcc" \
    # Pixi settings
    PIXI_HOME=/opt/pixi \
    PIXI_NO_PATH_UPDATE=1

# Core build deps only in builder
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates build-essential ninja-build ccache curl \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash && \
    ln -s /root/.pixi/bin/pixi /usr/local/bin/pixi

# Copy pixi configuration
COPY pixi.toml /tmp/ProteinDiff/pixi.toml

# Install dependencies with pixi (using cuda environment)
WORKDIR /tmp/ProteinDiff
RUN --mount=type=cache,target=/root/.cache/rattler,sharing=locked \
    pixi install --environment cuda && \
    pixi clean cache --yes

# Build FlashAttention wheel with robust caching
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.ccache \
    pixi run --environment cuda bash -c '\
      set -euo pipefail && \
      git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
      cd /tmp/flash-attention && \
      git reset --hard 5059fd5 && \
      python setup.py install && \
      rm -rf /tmp/flash-attention \
    '

################################
# Stage 2: runtime (CUDA slim) #
################################
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/opt/pixi \
    PIXI_NO_PATH_UPDATE=1

# Minimal runtime deps only
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates tini curl \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash && \
    ln -s /root/.pixi/bin/pixi /usr/local/bin/pixi

# Copy the pixi environment from builder
COPY --from=builder /tmp/ProteinDiff/.pixi /opt/ProteinDiff/.pixi
COPY --from=builder /tmp/ProteinDiff/pixi.toml /opt/ProteinDiff/pixi.toml
COPY --from=builder /tmp/ProteinDiff/pixi.lock /opt/ProteinDiff/pixi.lock

# Clean stray caches (usually none after builder clean, but safe)
RUN rm -rf /root/.cache /tmp/*

# Set up environment activation
ENV PATH="/opt/ProteinDiff/.pixi/envs/cuda/bin:$PATH"

# Non-root
RUN useradd -m -u 1000 -s /bin/bash coolguy && \
    chown -R coolguy:coolguy /opt/ProteinDiff
USER coolguy
WORKDIR /ProteinDiff

# (optional) tini for PID 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
