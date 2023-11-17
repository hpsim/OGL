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
*   _Ginkgo 1.5.0+_ (recommended to install via OGL)
*   C++17 compliant compiler (gcc or clang)

See also [ginkgo's](https://github.com/ginkgo-project/ginkgo) documentation for additional requirements.

![build](https://github.com/hpsim/OGL/actions/workflows/build-foam.yml/badge.svg)
![OF versions](https://img.shields.io/badge/OF--versions-v2212%2C10-green)
![Documentation](https://codedocs.xyz/hpsim/OGL/)

## Compilation

*OGL* can be build using cmake following the standard cmake procedure. 

    mkdir build && cd build && ccmake ..

By default *OGL* will fetch and build ginkgo, to specify which backend should be build you can use the following cmake flags `-DGINKGO_BUILD_CUDA`, `-DGINKGO_BUILD_OMP`, or ` -DGINKGO_BUILD_HIP`. For example to build *OGL* with *CUDA* and *OMP* support use

    cmake -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_OMP=ON ..

Then, compile and install by

    make -j && make install

### Ninja builds

If you have Ninja installed on your system we recommend to use ninja over gnu make for better compilation times 

    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DGINKGO_BUILD_HIP=ON  ../..
    cmake --build . --config Release
    cmake --install .

And make sure that the `system/controlDict` includes the `libOGL.so` or  `libOGL.dyLib` file:

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
* Multigrid (experimental)

additionally, the following preconditioners are available

### Supported Matrix Formats (Experimental)
Currently, the following matrix formats can be set by **matrixFormat**

* Coo 
* Csr
* Ell (not supported for ranksPerGPU != 1)
* Hybrid (not supported for ranksPerGPU != 1)

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


## Known Limitations

Currently, only basic cyclic boundary conditions are supported. Block-coupled matrices are not supported.

## Citing

When using OGL please cite the main Ginkgo paper describing Ginkgo's purpose, design and interface, which is
available through the following reference:

``` bibtex
@misc{anzt2020ginkgo,
    title={Ginkgo: A Modern Linear Operator Algebra Framework for High Performance Computing},
    author={Hartwig Anzt and Terry Cojean and Goran Flegar and Fritz Göbel and Thomas Grützmacher and Pratik Nayak and Tobias Ribizel and Yuhsiang Mike Tsai and Enrique S. Quintana-Ortí},
    year={2020},
    eprint={2006.16852},
    archivePrefix={arXiv},
    primaryClass={cs.MS}
}
```

## Example
Below an animation of a coarse 2D simulation of a karman vortex street performed on a MI100 can  be seen. Here both the momentum and Poisson equation are offloaded to the gpu.
[![karman](https://github.com/hpsim/OGL_DATA/blob/main/assets/U_mag_rainbow.gif)](https://github.com/hpsim/OGL_DATA/blob/main/assets/U_mag_rainbow.gif)
