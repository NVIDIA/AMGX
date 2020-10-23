#!/usr/bin/env sh
#
# Takes one optional argument with the name of the container to use.

set -ex

CONTAINERS=$(ls ci/containers)
if [ -n "${1}" ]; then
    CONTAINERS=$1
fi

if command -v shellcheck ; then
    shellcheck ci/*.sh
fi

for CONTAINER in $CONTAINERS; do
    BASE_NAME=$(basename "${CONTAINER}" .py)
    BASE_IMG="amgx:base_${BASE_NAME}"
    BUILD_DIR="build_${BASE_NAME}"
    RECIPE="ci/containers/${CONTAINER}"
    if ! test -f "${RECIPE}"; then
        echo "Container at \"${RECIPE}\" does not exist"
        exit 1
    fi
    hpccm --recipe "${RECIPE}" --format=docker | \
        docker build -t "${BASE_IMG}" -
    nvidia-docker \
        run \
        -v "$(pwd -LP)":/amgx \
        -u "$(id -u "${USER}")":"$(id -g "${USER}")" \
        "${BASE_IMG}" \
        bash -c "cd /amgx/ && ./ci/test.sh ${BUILD_DIR}"
done
