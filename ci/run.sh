#!/usr/bin/env sh

set -ex

mkdir -p ci/docker/cuda_10_2
hpccm --recipe ci/cuda_10_2.py --format=docker > ci/docker/cuda_10_2/Dockerfile
(
    cd ci/docker/cuda_10_2
    docker build -f Dockerfile -t amgx:cuda10.2 .
)
nvidia-docker \
    run \
    -v $(pwd -LP):/amgx \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    amgx:cuda10.2 bash -c "cd /amgx/ && ./ci/test.sh"
