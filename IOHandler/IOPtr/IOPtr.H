/*---------------------------------------------------------------------------*\
License
    This file is part of OGL.

    OGL is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
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

using mtx = gko::matrix::Csr<scalar>;
using val_array = gko::Array<scalar>;
using idx_array = gko::Array<label>;

template <class T>
class IOPtr : public regIOobject {
private:
    std::shared_ptr<T> ptr_;

public:
    // - Runtime type information
    TypeName("IOPtr");

    //- Construct from IOobject and a PtrList
    IOPtr(const IOobject &io, std::shared_ptr<T> in_ptr)
        : regIOobject(io), ptr_(in_ptr){};

    std::shared_ptr<T> get_ptr() { return ptr_; };

    bool writeData(Ostream &) const { return false; };
};

typedef IOPtr<gko::Executor> GKOExecPtr;
typedef IOPtr<gko::CudaExecutor> GKOCudaExecPtr;
typedef IOPtr<gko::ReferenceExecutor> GKOReferenceExecPtr;
typedef IOPtr<gko::OmpExecutor> GKOOmpExecPtr;
typedef IOPtr<gko::HipExecutor> GKOHipExecPtr;
typedef IOPtr<idx_array> GKOIDXIOPtr;
typedef IOPtr<gko::matrix::Csr<scalar>> GKOCSRIOPtr;
typedef IOPtr<gko::matrix::Dense<scalar>> GKOVECIOPtr;
typedef IOPtr<gko::preconditioner::Jacobi<>> GKOBJIOPtr;


}  // namespace Foam

#endif
