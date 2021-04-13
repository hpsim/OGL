# OpenFOAM Ginkgo Layer (OGL)
![build](https://github.com/greole/OGL/actions/workflows/cmake.yml/badge.svg)

A wrapper for [ginkgo](https://github.com/ginkgo-project/ginkgo) solver to provide GPGPU capabilities to [OpenFOAM](https://openfoam.org/) 

## Requirements

OGL has the following requirements

*   _cmake 3.9+_
*   _OpenFOAM 6+_
*   _Ginkgo 1.4.0+_
*   C++14 compliant compiler

See also [ginkgo's](https://github.com/ginkgo-project/ginkgo) documentation for additional requirements.

## Compilation


*OGL* can be build using cmake. Make sure that
the `system/controlDict` includes the `OGL.so` file:

    libs ("libOGL.so");
    

## Usage


OGL solver support the same syntax as the default *OpenFOAM* solver. Thus, to use a `CG` solver you can simply replace `PCG` by `GKOCG`. In order to run either with *CUDA*, *HIP*, or *OMP* support set the `executor` to `cuda`, `hip`, or `omp` in the  `system/fvSolution` dictionary. 

Currently the following solver are supported

* CG with and without block Jacobi preconditioner
* BiCGStab
* IR

## Known Limitations

Currently cyclic boundary conditions are not supported.

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
