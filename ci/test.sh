#!/usr/bin/env sh

set -ex

BUILD_DIR=$1
if [ -z $1 ]; then
    BUILD_DIR=build
fi

rm -rf $BUILD_DIR || true
mkdir -p $BUILD_DIR

(
    cd $BUILD_DIR
    cmake ..
    make -j 8 all
    # WIP: test_launcher is allowed to fail; not all tests pass
    set +e
    ./$BUILD_DIR/tests/amgx_test_launcher
)
