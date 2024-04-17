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
#pragma once

#include "DevicePersistent/Array/Array.H"
#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"
#include "DevicePersistent/Partition/Partition.H"
#include "MatrixWrapper/Combination/Combination.H"
#include "MatrixWrapper/LDUMatrix/HostMatrix.H"

#include "fvCFD.H"

namespace Foam {

struct lduMatrixInitFunctor {
    const objectRegistry &db_;

    const ExecutorHandler &exec_;

    const word field_name_;

    const label verbose_;

    const lduMatrix &matrix_;

    const FieldField<Field, scalar> &interfaceBouCoeffs_;

    const FieldField<Field, scalar> &interfaceIntCoeffs_;

    const lduInterfaceFieldPtrsList &interfaces_;

    const dictionary &solver_controls_;

    lduMatrixInitFunctor(const objectRegistry &db, const ExecutorHandler &exec,
                         const label verbose,
                         const word field_name,
                         const lduMatrix &matrix,
                         const FieldField<Field, scalar> &interfaceBouCoeffs,
                         const FieldField<Field, scalar> &interfaceIntCoeffs,
                         const lduInterfaceFieldPtrsList &interfaces,
                         const dictionary &solver_controls)
        : db_(db),
          exec_(exec),
          matrix_(matrix),
          field_name_(field_name),
          verbose_(verbose),
          interfaceBouCoeffs_(interfaceBouCoeffs),
          interfaceIntCoeffs_(interfaceIntCoeffs),
          interfaces_(interfaces),
          solver_controls_(solver_controls)
    {}

    void update(
        std::shared_ptr<HostMatrixWrapper> persistent_device_matrix) const
    {
        persistent_device_matrix->update(
            matrix_.diag().begin(), matrix_.upper().begin(),
            matrix_.lower().begin(), interfaceBouCoeffs_, interfaceIntCoeffs_,
            interfaces_);
    }

    std::shared_ptr<HostMatrixWrapper> init() const
    {
        return HostMatrixWrapper::create(
            exec_, db_, matrix_.diag().size(), matrix_.upper().size(),
            matrix_.symmetric(), matrix_.diag().begin(),
            matrix_.upper().begin(), matrix_.lower().begin(),
            matrix_.lduAddr(), interfaceBouCoeffs_,
            interfaceIntCoeffs_, interfaces_, solver_controls_, field_name_,
            verbose_);
    }
};


/* This Class makes the Host LDUMatrix device persistent. It basically only
 * provides a functor to call HostMatrixWrapper::create and LduMatrix::update
 */
class PersistentHostMatrixAdapter {
private:
    const objectRegistry &db_;

    const label verbose_;

    const word field_name_;

    mutable PersistentBase<HostMatrixWrapper, lduMatrixInitFunctor> host_matrix_;


public:
    template<typename matrixType>
    PersistentHostMatrixAdapter(const objectRegistry &db, const ExecutorHandler &exec,
                  const dictionary &controlDict, const word field_name,
                  const label verbose,
                  const matrixType& matrix,
                  const FieldField<Field, scalar> &interfaceBouCoeffs,
                  const FieldField<Field, scalar> &interfaceIntCoeffs,
                  const lduInterfaceFieldPtrsList &interfaces)
        : db_(db),
          verbose_(verbose),
          field_name_(field_name),
          host_matrix_{
              field_name_ + "_host_matrix", db,
              lduMatrixInitFunctor(db, exec, verbose_,
                                   field_name + "_host_matrix",
                                   matrix,
                                   interfaceBouCoeffs, interfaceIntCoeffs,
                                   interfaces, controlDict),
              controlDict.lookupOrDefault<Switch>("updateSysMatrix", true),
              verbose_}
    {}

    std::shared_ptr<HostMatrixWrapper> get() const
    {
        return host_matrix_.get_persistent_object();
    }
};

}  // namespace Foam
