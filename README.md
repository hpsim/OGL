# OpenFOAM Ginkgo Layer (OGL)
A wrapper for [ginkgo](https://github.com/ginkgo-project/ginkgo) solvers and preconditioners to provide GPGPU capabilities to OpenFOAM.

> [Requirements](https://github.com/hpsim/OGL#requirements)<br/>
> [Compilation](https://github.com/hpsim/OGL#Compilation)<br/>
> [Usage](https://github.com/hpsim/OGL#Usage)<br/>
> [Known Limitations](https://github.com/hpsim/OGL#Known_Limitations)<br/>
> [Citing](https://github.com/hpsim/OGL#Citing)<br/>
> [Example](https://github.com/hpsim/OGL#Example)<br/>
> [Performance](https://github.com/hpsim/OGL#Performance)<br/>


## Requirements

OGL has the following requirements

*   _cmake 3.9+_
*   _OpenFOAM 6+_ or _v2106_
*   _Ginkgo 1.4.0+_
*   C++14 compliant compiler (gcc or clang)

See also [ginkgo's](https://github.com/ginkgo-project/ginkgo) documentation for additional requirements.

![ESI OpenFOAM](https://github.com/hpsim/OGL/actions/workflows/build-esi.yml/badge.svg)
![ESI OpenFOAM](https://github.com/hpsim/OGL/actions/workflows/build-extend.yml/badge.svg)
![ESI OpenFOAM](https://github.com/hpsim/OGL/actions/workflows/build.yml/badge.svg)

## Compilation

*OGL* can be build using cmake following the standard cmake procedure. 

    mkdir build && cd build && ccmake ..

By default *OGL* will fetch and build ginkgo, to specify which backend should be build you can use the following cmake flags `-DGINKGO_BUILD_CUDA`, `-DGINKGO_BUILD_OMP`, or ` -DGINKGO_BUILD_HIP`. For example to build *OGL* with *CUDA* and *OMP* support use

    cmake -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_OMP=ON ..

Then, make sure that the `system/controlDict` includes the `libOGL.so` or  `libOGL.dyLib` file:

    libs ("libOGL.so");

### Experimental OGL ginkgo features

Some of OGL features might depend on features which are not already implemented on ginkgo's dev branch. To enable experimental features pass `-DGINKGO_WITH_OGL_EXTENSIONS` as cmake flag.


## Usage

OGL solver support the same syntax as the default *OpenFOAM* solver. Thus, to use a `CG` solver you can simply replace `PCG` by `GKOCG`. In order to run either with *CUDA*, *HIP*, or *OMP* support set the `executor` to `cuda`, `hip`, or `omp` in the  `system/fvSolution` dictionary.

Currently, the following solver are supported

* CG
* BiCGStab
* GMRES
* IR (experimental)
* Multigrid (experimental)

additionally, the following preconditioner are availible

* BJ, block Jacobi
* ILU, incomplete LU (experimental)
* IC, incomplete Cholesky (experimental)
* ISAI (experimental)
* Multigrid, algebraic multigrid (experimental)

The following optional solver arguments are supported

Argument | Default | Description
------------ | ------------- | -------------
updateSysMatrix | true | whether to copy the system matrix to device on every solver call
updateInitVector | false |whether to copy the initial guess to device on every solver call
sort | true | sort the system matrix
executor | reference | the executor where to solve the system matrix, other options are `omp`, `cuda`
export | false | write the complete system to disk
verbose | false | print out extra info
evalFrequency | 1 | evaluate residual norm every n-th iteration
adaptMinIter | true | based on the previous solution set minIter to be relaxationFactor*previousIters
relaxationFactor | 0.8 | use relaxationFactor*previousIters as new minIters

## Known Limitations

Currently cyclic boundary conditions and coupled matrices are not supported.

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

## Performance
[![Performance](https://img.shields.io/badge/Performance-Data-brightgreen)](https://github.com/greole/OGL_DATA)

A detailed overview of performance data is given in a separate  [data repository](https://github.com/greole/OGL_DATA).
