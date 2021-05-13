#!/bin/bash

if [ $EXTERNAL_GINKGO = ON ]
then
    mkdir -p $HOME/cache/
    cd $HOME/cache
    git pull -C ginkgo-$GINKGO_VERSION pull ||             \
    git clone https://github.com/ginkgo-project/ginkgo.git \
        $HOME/cache/ginkgo-$GINKGO_VERSION
    mkdir -p $HOME/cache/ginkgo-$GINKGO_VERSION/build
    cd $HOME/cache/ginkgo-$GINKGO_VERSION/build
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
    make  -j4 &&  sudo make install
fi
