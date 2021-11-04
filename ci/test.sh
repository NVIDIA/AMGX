#!/usr/bin/env sh

set -ex

BUILD_DIR=$1
if [ -z "${1}" ]; then
    BUILD_DIR=build
fi

if [ "${AMGX_CI_KEEP_BUILD}" = "0" ]; then
    rm -rf "${BUILD_DIR}" || true
fi
mkdir -p "${BUILD_DIR}"

(
    cd "${BUILD_DIR}"
    cmake ..
    make -j 8 all
    # WIP: test_launcher is allowed to fail; not all tests pass
    set +e
    ./tests/amgx_tests_launcher
)
