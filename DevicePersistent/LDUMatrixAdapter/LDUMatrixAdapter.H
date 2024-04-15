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
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/
#ifndef OGL_IOGKOMatrixHandler_INCLUDED_H
#define OGL_IOGKOMatrixHandler_INCLUDED_H


#include "DevicePersistent/Array/Array.H"
#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"
#include "DevicePersistent/Partition/Partition.H"

#include "MatrixWrapper/Combination/Combination.H"
#include "MatrixWrapper/Distributed/Distributed.H"

#include "fvCFD.H"

namespace Foam {

struct lduMatrixInitFunctor {

    const objectRegistry &db_;

    const ExecutorHandler &exec_;

    const lduMatrix & matrix_;

    const FieldField<Field, scalar> &interfaceBouCoeffs_;

    const FieldField<Field, scalar> &interfaceIntCoeffs_;

    const lduInterfaceFieldPtrsList &interfaces_;

    MatrixInitFunctor(const objectRegistry &db, const ExecutorHandler &exec,
                      const lduMatrix & matrix,
                      const word field_name, const label verbose,
                      const FieldField<Field, scalar> &interfaceBouCoeffs,
                      const FieldField<Field, scalar> &interfaceIntCoeffs,
                      const lduInterfaceFieldPtrsList &interfaces)
        : db_(db),
          exec_(exec),
          matrix_(matrix),
          field_name_(field_name) verbose_(verbose),
          interfaceBouCoeffs_(interfaceBouCoeffs),
        interfaceIntCoeffs_(interfaceIntCoeffs),
        interfaces_(interfaces)
    {}

    void update(std::shared_ptr<LduMatrix> persistent_device_matrix) const
    {
        persistent_device_matrix.update(exec_, interfaceBouCoeffs_,
                                        interfaceIntCoeffs_, interfaces_);
    }

    std::shared_ptr<LduMatrix> init() const
    {
        return LduMatrix::create(
          exec_,
          matrix_.diag().size(),
          matrix_.lduAddr().upperAddr().size(),
          matrix_.symmetric(),
          matrix_.diag().begin(),
          matrix_.upper().begin(),
          matrix_.lower().begin(),
          matrix_.lduAddr().upperAddr().begin(),
          matrix_.lduAddr().lowerAddr().begin(),
          interfaceBouCoeffs_,
          interfaceIntCoeffs_,
          interfaces_);
    }
};


/* This Class makes the Host LDUMatrix device persistent. It basically only
 * provides a functor to call LduMatrix::create and LduMatrix::update
 */
class MatrixWrapper {
private:
    const objectRegistry &db_;

    const label verbose_;

    const word field_name_;

    mutable PersistentBase<LDUMatrix, MatrixInitFunctor> host_matrix_;


public:
    MatrixWrapper(const objectRegistry &db, const PersistentExecutor &exec,
                  const dictionary &controlDict, const word sys_matrix_name,
                  const label verbose,
                  const FieldField<Field, scalar> &interfaceBouCoeffs,
                  const FieldField<Field, scalar> &interfaceIntCoeffs,
                  const lduInterfaceFieldPtrsList &interfaces)
        : db_(db),
          verbose_(verbose),
          export_(controlDict.lookupOrDefault<Switch>("export", false)),
          field_name_(sys_matrix_name),
          host_matrix_{
              sys_matrix_name + "_host_matrix", db,
              MatrixInitFunctor(
                  db, exec, verbose_, sys_matrix_name + "_host_matrix",
                  interfaceBouCoeffs, interfaceIntCoeffs, interfaces),
              controlDict.lookupOrDefault<Switch>("updateSysMatrix", true),
              verbose_}
    {}

    std::shared_ptr<gko::LinOp> get() const
    {
        return gkomatrix_.get_persistent_object();
    }
};

}  // namespace Foam

#endif
