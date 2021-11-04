#!/usr/bin/env sh
#
# Supports following environment variables:
#
# AMGX_CI_CONTAINERS: list of containers to run. Default: all containers.
#
# AMGX_CI_KEEP_BUILD: preserves build directory. Default: build directories
# are cleaned up each time.
set -ex

if command -v shellcheck ; then
    shellcheck ci/*.sh
fi

CONTAINERS=$(ls ci/containers)
if [ -n "${AMGX_CI_CONTAINERS}" ]; then
    CONTAINERS="${AMGX_CI_CONTAINERS}"
fi

KEEP_BUILD=0
if [ -n "${AMGX_CI_KEEP_BUILD}" ]; then
    KEEP_BUILD="${AMGX_CI_KEEP_BUILD}"
fi

CONTAINER_FILE=/dev/fd/2

for CONTAINER in $CONTAINERS; do
    BASE_NAME=$(basename "${CONTAINER}" .py)
    BASE_IMG="amgx:base_${BASE_NAME}"
    BUILD_DIR="build_${BASE_NAME}"
    RECIPE="ci/containers/${CONTAINER}"
    if ! test -f "${RECIPE}"; then
        echo "Container at \"${RECIPE}\" does not exist"
        exit 1
    fi

    if [ -n "${AMGX_CI_CONTAINER_FILE}" ]; then
        CONTAINER_FILE="Dockerfile_${BASE_NAME}"
    fi

    hpccm --recipe "${RECIPE}" --format=docker \
        | tee "${CONTAINER_FILE}" | \
        docker build -t "${BASE_IMG}" -
    nvidia-docker \
        run \
        -v "$(pwd -LP)":/amgx \
        -u "$(id -u "${USER}")":"$(id -g "${USER}")" \
        "${BASE_IMG}" \
        bash -c "cd /amgx/ && AMGX_CI_KEEP_BUILD=${KEEP_BUILD} ./ci/test.sh ${BUILD_DIR}"
done
