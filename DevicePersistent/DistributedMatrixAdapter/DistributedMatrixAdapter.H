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
#include "MatrixWrapper/Distributed/Distributed.H"

#include "fvCFD.H"

namespace Foam {

struct DistributedMatrixInitFunctor {
    const objectRegistry &db_;

    const ExecutorHandler &exec_;

    const word field_name_;

    const label verbose_;

    const dictionary& solver_controls_;

    const PersistentHostMatrixAdapter& host_A_;

    DistributedMatrixInitFunctor(const objectRegistry &db, const ExecutorHandler &exec,
                         const label verbose,
                         const word field_name,
                         const PersistentHostMatrixAdapter& host_A,
                         const dictionary& solver_controls
                         )
        : db_(db),
          exec_(exec),
          field_name_(field_name),
          verbose_(verbose),
          solver_controls_(solver_controls),
          host_A_(host_A)
    {}

    void update(
        std::shared_ptr<RepartDistMatrix<scalar, label, label>> persistent_device_matrix) const
    {
        // persistent_device_matrix->update();
    }

    std::shared_ptr<RepartDistMatrix<scalar, label, label>> init() const
    {
        return RepartDistMatrix<scalar, label, label>::create(
              solver_controls_.lookupOrDefault<word>("matrixFormat", "Coo"),
              host_A_.get());
    }
};


/* This Class makes the Host LDUMatrix device persistent. It basically only
 * provides a functor to call HostMatrixWrapper::create and LduMatrix::update
 */
class PersistentDistributedMatrix {
private:
    const objectRegistry &db_;

    const label verbose_;

    const word field_name_;

    mutable PersistentBase<RepartDistMatrix<scalar, label, label>, DistributedMatrixInitFunctor> host_matrix_;


public:
    PersistentDistributedMatrix(
                  const objectRegistry &db,
                  const ExecutorHandler &exec,
                  const word field_name,
                  const PersistentHostMatrixAdapter& host_A,
                  const dictionary &controlDict,
                  const label verbose
                  )
        : db_(db),
          verbose_(verbose),
          field_name_(field_name),
          host_matrix_{
              field_name + "_distributed_matrix", db,
              DistributedMatrixInitFunctor(db, exec, verbose_,
                                   field_name + "_distributed_matrix",
                                   host_A, 
                                   controlDict),
              controlDict.lookupOrDefault<Switch>("updateSysMatrix", true),
              verbose_}
    {}

    std::shared_ptr<RepartDistMatrix<scalar, label, label>> get() const
    {
        return host_matrix_.get_persistent_object();
    }

    void write() 
    {
        this->get()->write();
    }

    bool get_export();
};

}  // namespace Foam
