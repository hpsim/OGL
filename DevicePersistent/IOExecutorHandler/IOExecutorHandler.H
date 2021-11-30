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

#ifndef OGL_IOExecutorHandler_INCLUDED_H
#define OGL_IOExecutorHandler_INCLUDED_H

#include "../IOPtr/IOPtr.H"
#include "fvCFD.H"
#include "regIOobject.H"

#include <ginkgo/ginkgo.hpp>

namespace Foam {

class IOExecutorHandler {
private:
    const word device_executor_name_;

    const word app_executor_name_;

    // executor where Ginkgo will perform the computation
    GKOExecPtr *device_exec_ptr_;

    // executor of the application
    GKOExecPtr *app_exec_ptr_;

    GKOReferenceExecPtr *ref_exec_ptr_;

    GKOCudaExecPtr *cuda_exec_ptr_;

    GKOOmpExecPtr *omp_exec_ptr_;

    GKOHipExecPtr *hip_exec_ptr_;

    GKODpcppExecPtr *dpcpp_exec_ptr_;

public:
    IOExecutorHandler(const objectRegistry &db, const dictionary &controlDict);

    word get_device_executor_name() const { return device_executor_name_; }

    std::shared_ptr<gko::Executor> get_device_executor() const
    {
        const word device_executor_name{get_device_executor_name()};
        if (device_executor_name == "omp") {
            return omp_exec();
        }
        if (device_executor_name == "cuda") {
            return cuda_exec();
        }
        if (device_executor_name == "hip") {
            return hip_exec();
        }
        return ref_exec();
    };

    std::shared_ptr<gko::Executor> app_exec() const
    {
        return app_exec_ptr_->get_ptr();
    };

    std::shared_ptr<gko::OmpExecutor> omp_exec() const
    {
        return omp_exec_ptr_->get_ptr();
    };

    std::shared_ptr<gko::CudaExecutor> cuda_exec() const
    {
        return cuda_exec_ptr_->get_ptr();
    };

    std::shared_ptr<gko::HipExecutor> hip_exec() const
    {
        return hip_exec_ptr_->get_ptr();
    };

    std::shared_ptr<gko::ReferenceExecutor> ref_exec() const
    {
        return ref_exec_ptr_->get_ptr();
    };
};

}  // namespace Foam
#endif
