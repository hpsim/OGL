# OGL

A wrapper for [Ginkgo](https://github.com/ginkgo-project/ginkgo) solver to provide GPGPU capabilities to [OpenFOAM](https://openfoam.org/) 

# Requirements

OGL has the following requirements

* Ginkgo
* OpenFOAM

# Compilation

OGL can be compiled using "wmake" in the main folder. Make sure that
the "controlDict" includes the "OGL.so" file:

    libs ("libOGL.so");
    
and that "libGinkgo.so" can be found.

# Usage

OGL solver support the same syntax as the default OpenFOAM solver. Thus, to use a CG solver you can simply replace "PCG" by "GKOCG". In order to run either with CUDA, HIP, or OMP support set the "executor" to cuda, hip, or omp in your solver dictionary. 


