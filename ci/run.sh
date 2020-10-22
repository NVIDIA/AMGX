#!/usr/bin/env sh

set -ex

CONTAINERS="\
x86_64-ubuntu18.04-llvm-cuda11.0 \
x86_64-ubuntu18.04-gnu-cuda11.0 \
x86_64-ubuntu18.04-gnu-cuda10.2 \
"

for CONTAINER in $CONTAINERS; do
    BASE_IMG="amgx:base_${CONTAINER}"
    BUILD_DIR="build_${CONTAINER}"
    hpccm --recipe ci/containers/$CONTAINER.py --format=docker | \
        docker build -t $BASE_IMG -
    nvidia-docker \
        run \
        -v $(pwd -LP):/amgx \
        -u $(id -u ${USER}):$(id -g ${USER}) \
        $BASE_IMG \
        bash -c "cd /amgx/ && ./ci/test.sh ${BUILD_DIR}"
done
