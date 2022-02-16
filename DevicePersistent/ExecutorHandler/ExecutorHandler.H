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
    Foam::ExecutorHandler

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    ExecutorHandler.C

\*---------------------------------------------------------------------------*/

#ifndef OGL_ExecutorHandler_INCLUDED_H
#define OGL_ExecutorHandler_INCLUDED_H

#include "../DevicePersistentBase/DevicePersistentBase.H"
#include "fvCFD.H"

#include <ginkgo/ginkgo.hpp>

namespace Foam {

struct ExecutorInitFunctor {
    mutable std::shared_ptr<gko::mpi::communicator> comm_;

    const label device_id_;

    const word executor_name_;

    const word field_name_;

    const label verbose_;

    ExecutorInitFunctor(const word executor_name, const word field_name,
                        const label verbose)
        : comm_(std::make_shared<gko::mpi::communicator>(MPI_COMM_WORLD)),
          device_id_(comm_->node_local_rank()),
          executor_name_(executor_name),
          field_name_(field_name),
          verbose_(verbose)
    {}


    void update(std::shared_ptr<gko::Executor> e) const {}

    std::shared_ptr<gko::Executor> init() const
    {
        auto host_exec = gko::share(gko::ReferenceExecutor::create());

        if (executor_name_ == "cuda") {
            return gko::share(
                gko::CudaExecutor::create(device_id_, host_exec, true));
        }
        if (executor_name_ == "dpcpp") {
            return gko::share(
                gko::DpcppExecutor::create(device_id_, host_exec));
        }
        if (executor_name_ == "hip") {
            return gko::share(
                gko::HipExecutor::create(device_id_, host_exec, true));
        }
        if (executor_name_ == "omp") {
            return gko::share(gko::OmpExecutor::create());
        }
        if (executor_name_ == "reference") {
            return host_exec;
        }

        // NOTE Return empty pointer to avoid compiler warnings
        return {};
    }
};

class ExecutorHandler
    : public PersistentBase<gko::Executor, ExecutorInitFunctor> {
private:
    mutable std::shared_ptr<gko::mpi::communicator> comm_;

    const word device_executor_name_;

public:
    ExecutorHandler(const objectRegistry &db, const dictionary &solverControls,
                    const word field_name)
        : PersistentBase<gko::Executor, ExecutorInitFunctor>(
              solverControls.lookupOrDefault("executor", word("reference")) +
                  +"_" + field_name,
              db,
              ExecutorInitFunctor(
                  solverControls.lookupOrDefault("executor", word("reference")),
                  field_name,
                  solverControls.lookupOrDefault("verbose", label(0))),
              true, 0),
          comm_(std::make_shared<gko::mpi::communicator>(MPI_COMM_WORLD)),
          device_executor_name_(
              solverControls.lookupOrDefault("executor", word("reference")))
    {}

    const std::shared_ptr<gko::Executor> get_device_exec() const
    {
        return this->get_persistent_object();
    }

    const std::shared_ptr<gko::Executor> get_ref_exec() const
    {
        return get_device_exec()->get_master();
    }

    word get_exec_name() const { return device_executor_name_; }

    std::shared_ptr<gko::mpi::communicator> get_gko_mpi_comm_wrapper() const
    {
        return this->comm_;
    }
};

using PersistentExecutor = ExecutorHandler;

}  // namespace Foam
// namespace Foam
#endif
