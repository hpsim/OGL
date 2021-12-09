/*---------------------------------------------------------------------------*\
License
    This file is part of OGL.

    OGL is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OGL is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OGL.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::IOExecutorHandler

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    IOExecutorHandler.C

\*---------------------------------------------------------------------------*/

#ifndef OGL_IOPtr_INCLUDED_H
#define OGL_IOPtr_INCLUDED_H

#include <ginkgo/ginkgo.hpp>
#include "fvCFD.H"


namespace Foam {

/* Base class for storing ginkgo and OGL data structures in the Objectregistry
 *
 * On creation it registers it self to the object registry and holds
 * a shared_ptr<T> where T is a gko data type
 *  */
template <class T>
class DevicePersistentBase : public regIOobject {
private:
    std::shared_ptr<T> ptr_;

public:
    // - Runtime type information
    TypeName("IOPtr");

    //- Construct from IOobject and a PtrList
    DevicePersistentBase(const IOobject &io, std::shared_ptr<T> in_ptr)
        : regIOobject(io), ptr_(in_ptr){};

    // get a ptr to the underlying data
    std::shared_ptr<T> get_ptr() { return ptr_; };

    bool writeData(Ostream &) const { return false; };
};


    // TODO is that needed?
typedef DevicePersistentBase<gko::Executor>  GKOExecPtr;
typedef DevicePersistentBase<gko::CudaExecutor> GKOCudaExecPtr;
typedef DevicePersistentBase<gko::ReferenceExecutor> GKOReferenceExecPtr;
typedef DevicePersistentBase<gko::OmpExecutor> GKOOmpExecPtr;
typedef DevicePersistentBase<gko::HipExecutor> GKOHipExecPtr;
typedef DevicePersistentBase<gko::DpcppExecutor> GKODpcppExecPtr;
// typedef DevicePersistentBase<gko::matrix::Csr<scalar>> GKOCSRIOPtr;
// typedef DevicePersistentBase<gko::matrix::Dense<scalar>> GKOVECIOPtr;
// typedef DevicePersistentBase<gko::preconditioner::Jacobi<>> GKOBJIOPtr;
// typedef DevicePersistentBase<gko::preconditioner::Ilu<>> GKOILUIOPtr;
// typedef DevicePersistentBase<gko::LinOp> GKOLinOPIOPtr;


}  // namespace Foam

#endif
