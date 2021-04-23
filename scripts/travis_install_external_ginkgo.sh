#!/bin/bash


if [ $EXTERNAL_GINKGO = ON ]
then
    git clone https://github.com/ginkgo-project/ginkgo.git
    mkdir ginkgo/build
    cd ginkgo/build
    cmake -DGINKGO_BUILD_BENCHMARKS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_OMP=OFF ..
    make  && make install
fi
