# syntax=docker/dockerfile:1.6

# setup nvidia env with micromamba
FROM mambaorg/micromamba:1.5.8 AS mmb
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

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

# edit paths
ENV PATH=/opt/conda/envs/ProteinDiff_env/bin:$PATH

# no root user
RUN useradd -m -u 1000 -s /bin/bash coolguy
USER coolguy

WORKDIR /ProteinDiff/
