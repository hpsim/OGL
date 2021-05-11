#!/bin/bash

if [ $EXTERNAL_GINKGO = ON ]
then
    git clone https://github.com/ginkgo-project/ginkgo.git
    mkdir ginkgo/build
    cd ginkgo/build
    git checkout $GINKGO_VERSION
    cmake \
        -DGINKGO_BUILD_BENCHMARKS=OFF \
        -DGINKGO_BUILD_EXAMPLES=OFF \
        -DGINKGO_BUILD_CUDA=off \
        -DGINKGO_BUILD_HIP=off \
        -DGINKGO_BUILD_OMP=off \
        -DGINKGO_BUILD_TESTS=off\
        -DGINKGO_BUILD_REFERENCE=on \
        -DCMAKE_BUILD_TYPE=Release \
        -DGINKGO_BUILD_HWLOC=off \
        ..
    make  &&  sudo make install
fi
