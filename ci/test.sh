#!/usr/bin/env sh

set -ex

rm -rf build || true
mkdir -p build

(
    cd build
    cmake ..
    make -j 16 all
    ./build/tests/test_launcher
)
