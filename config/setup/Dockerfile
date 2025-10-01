# syntax=docker/dockerfile:1.6

# setup nvidia env with micromamba
FROM mambaorg/micromamba:1.5.8 AS mmb
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS runtime

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
# Notes:
#  - use --no-build-isolation so it uses the env's torch/pybind/etc.
#  - FA3 build system uses setup.py in the hopper/ folder.
RUN --mount=type=cache,target=/root/.cache/pip \
    micromamba run -n ProteinDiff_env bash -lc '\
      python -m pip install --upgrade pip setuptools wheel && \
      git clone --depth 1 --branch "${FA3_REF}" https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
      cd /tmp/flash-attention/hopper && \
      python setup.py install \
    ' && \
    rm -rf /tmp/flash-attention

# edit paths
ENV PATH=/opt/conda/envs/ProteinDiff_env/bin:$PATH

# no root user
RUN useradd -m -u 1000 -s /bin/bash coolguy
USER coolguy

WORKDIR /ProteinDiff/

# install flash attention (for hopper)
RUN git clone https://github.com/Dao-AILab/flash-attention.git@5059fd53e602bcc00336bb5cc8a85e50940485cb && \
    cd hopper && \
    python setup.py install && \
    