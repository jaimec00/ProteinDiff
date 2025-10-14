#!/bin/bash

# this seems a bit obtuse, so putting it all here so user just runs ./config/setup/start.sh train

# get compute capability
NVIDIA_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '{print $1}')
REMOTE=jaimec00/proteindiff:nvidia12.8-sm${NVIDIA_SM}
LOCAL=proteindiff:dev

if ! docker image inspect $LOCAL >/dev/null 2>&1; then
    # pull from docker (sm 80 available, sm 90 not built yet)
    sudo docker pull $REMOTE
    sudo docker tag $REMOTE $LOCAL
fi

sudo docker compose -f config/setup/docker-compose.yml --env_file config/setup/.env run --rm $1
