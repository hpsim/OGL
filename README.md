**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
**[Usage](#Usage)** |
**[Known Limitations](#Known_Limitations)** |
**[Citing](#Citing)** |
**[Example](#Example)** |
**[Performance](#Performance)** 

---

OGL is a wrapper for [ginkgo](https://github.com/ginkgo-project/ginkgo) solvers and preconditioners to provide GPGPU capabilities to OpenFOAM.


## Requirements

OGL has the following requirements

*   _cmake 3.13+_
*   _OpenFOAM 6+_ or _v2106_
*   _Ginkgo 1.5.0+_ (It is recommended to install via OGL)
*   C++17 compliant compiler (gcc or clang)

See also [ginkgo's](https://github.com/ginkgo-project/ginkgo) documentation for additional requirements.

![build](https://github.com/hpsim/OGL/actions/workflows/build-foam.yml/badge.svg)
![OF versions](https://img.shields.io/badge/OF--versions-v2212%2C10-green)
![Documentation](https://codedocs.xyz/hpsim/OGL/)

For cuda builds cuda version 12 is recommended. For older cuda versions automatic device detection might fail, in this case please set the cuda architecture manually via `-DOGL_CUDA_ARCHITECTURES`.

## Compilation

*OGL* can be build using cmake following the standard cmake procedure. 

    mkdir build && cd build && ccmake ..

By default *OGL* will fetch and build ginkgo, to specify which backend should be build you can use the following cmake flags `-DGINKGO_BUILD_CUDA`, `-DGINKGO_BUILD_OMP`, or ` -DGINKGO_BUILD_HIP`. For example to build *OGL* with *CUDA* and *OMP* support use

    cmake -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_OMP=ON ..

Then, compile and install by

    make -j && make install

### CMakePresets and Ninja builds

If you have Ninja installed on your system we recommend to use ninja over gnu make for better compilation times. We also provide a list of Cmake presets which can be used a recent version of Cmake (>3.20). To display available presets use: 

    cmake --list-preset
    
The following example shows how to execute a build and install on a cuda system.

    cmake --preset ninja-cuda-release
    cmake --build --preset ninja-cpuonly-release  --target install


After a sucesfull build install make sure that the `system/controlDict` includes the `libOGL.so` or  `libOGL.dyLib` file:

    libs ("libOGL.so");


## Usage

OGL solver support the same syntax as the default *OpenFOAM* solver. Thus, to use Ginkgo's `CG` solver you can simply replace `PCG` by `GKOCG`. In order to run either with *CUDA*, *HIP*, or *OMP* support set the `executor` keyword to `cuda`, `hip`, or `omp` in the  `system/fvSolution` dictionary.

Argument | Default | Description
------------ | ------------- | -------------
ranksPerGPU  | 1 | gather from n ranks to GPU
updateRHS | true | whether to copy the system matrix to device on every solver call
updateInitGuess | false |whether to copy the initial guess to device on every solver call
export | false | write the complete system to disk
verbose | 0 | print out extra info
executor | reference | the executor where to solve the system matrix, other options are `omp`, `cuda`
adaptMinIter | true | based on the previous solution set minIter to be relaxationFactor*previousIters
relaxationFactor | 0.8 | use relaxationFactor*previousIters as new minIters
scaling | 1.0 | Scale the complete system by the scaling factor
forceHostBuffer  | false | whether to copy to host before MPI calls

### Supported Solver
Currently, the following solver are supported

* CG
* BiCGStab
* GMRES
* IR (experimental)

additionally, the following preconditioners are available

### Supported Preconditioner
* BJ, block Jacobi
* [ISAI](https://doi.org/10.1016/j.parco.2017.10.003), Incomplete Sparse Approximate Inverses,
* ILU, incomplete LU (experimental)
* IC, incomplete Cholesky (experimental)
* Multigrid, algebraic multigrid (experimental)

The following optional arguments are supported to modify the preconditioner. *Note* some preconditioners like IC or (SPD) ISAI require positive values on the system matrix diagonal, thus in case of the pressure equation the complete system needs to be scaled by a factor of -1.0.

Argument | Default | Preconditioner
------------ | ------------- | -------------
SkipSorting | True | all
Caching | 1 | all
MaxBlockSize | 1 | block Jacobi 
SparsityPower | 1 | ISAI
MaxLevels | 9 | Multigrid
MinCoarseRows | 10 | Multigrid
ZeroGuess | True | Multigrid

### Supported Matrix Formats (Experimental)
Currently, the following matrix formats can be set by **matrixFormat**

* Coo 
* Csr
* Ell (not supported for ranksPerGPU != 1)
* Hybrid (not supported for ranksPerGPU != 1)


## Known Limitations and Troubleshooting

- Currently, only basic cyclic boundary conditions are supported, no AMI boundary conditions are supported. Block-coupled matrices are not supported.

- If you are compiling against a double precision label version of OpenFOAM 
make sure to set `-DOGL_DP_LABELS=ON` otherwise errors of the following type can occur  `undefined symbol: _ZN4Foam10dictionary3addERKNS_7keyTypeEib`

## Citing

When using OGL please cite the main Ginkgo paper describing Ginkgo's purpose, design and interface, which is
available through the following reference:

``` bibtex
@article{Anzt_Ginkgo_A_Modern_2022,
author = {Anzt, Hartwig and Cojean, Terry and Flegar, Goran and Göbel, Fritz and Grützmacher, Thomas and Nayak, Pratik and Ribizel, Tobias and Tsai, Yuhsiang and Quintana-Ortí, Enrique S.},
doi = {10.1145/3480935},
journal = {ACM Transactions on Mathematical Software},
month = mar,
number = {1},
pages = {1--33},
title = {{Ginkgo: A Modern Linear Operator Algebra Framework for High Performance Computing}},
volume = {48},
year = {2022}
}
```

## Example
Below an animation of a coarse 2D simulation of a karman vortex street performed on a MI100 can be seen. Here both the momentum and Poisson equation are offloaded to the GPU.
[![karman](https://github.com/hpsim/OGL_DATA/blob/main/assets/U_mag_rainbow.gif)](https://github.com/hpsim/OGL_DATA/blob/main/assets/U_mag_rainbow.gif)
