
#include <ginkgo/ginkgo.hpp>
#include <type_traits>

#include "IOExecutorHandler.H"

namespace Foam {

IOExecutorHandler::IOExecutorHandler(const objectRegistry &db,
                                     const dictionary &controlDict)
    : device_executor_name_(
          controlDict.lookupOrDefault("executor", word("reference"))),
      app_executor_name_(
          controlDict.lookupOrDefault("app_executor", word("reference")))
{
    // create executors
    bool app_exec_stored = db.foundObject<regIOobject>(app_executor_name_);

    if (app_exec_stored) {
        ref_exec_ptr_ =
            &db.lookupObjectRef<GKOReferenceExecPtr>(app_executor_name_);
        if (device_executor_name_ == app_executor_name_) {
            return;
        }
    }


    bool device_exec_stored =
        db.foundObject<regIOobject>(device_executor_name_);
    if (device_exec_stored) {
        if (device_executor_name_ == "omp") {
            omp_exec_ptr_ =
                &db.lookupObjectRef<GKOOmpExecPtr>(device_executor_name_);
            return;
        }
        if (device_executor_name_ == "cuda") {
            cuda_exec_ptr_ =
                &db.lookupObjectRef<GKOCudaExecPtr>(device_executor_name_);
            return;
        }
        if (device_executor_name_ == "hip") {
            hip_exec_ptr_ =
                &db.lookupObjectRef<GKOHipExecPtr>(device_executor_name_);
            return;
        }
        if (device_executor_name_ == "dpcpp") {
            dpcpp_exec_ptr_ =
                &db.lookupObjectRef<GKODpcppExecPtr>(device_executor_name_);
            return;
        }
    }

    const fileName app_exec_store = app_executor_name_;
    ref_exec_ptr_ =
        new GKOReferenceExecPtr(IOobject(app_exec_store, db),
                                gko::give(gko::ReferenceExecutor::create()));

    const fileName device_exec_store = device_executor_name_;
    const fileName omp_exec_store = "omp";
    omp_exec_ptr_ = new GKOOmpExecPtr(IOobject(omp_exec_store, db),
                                      gko::OmpExecutor::create());

    if (device_executor_name_ == "cuda") {
        cuda_exec_ptr_ =
            new GKOCudaExecPtr(IOobject(device_exec_store, db),
                               gko::give(gko::CudaExecutor::create(
                                   0, omp_exec_ptr_->get_ptr(), true)));
    }
    if (device_executor_name_ == "dpcpp") {
        dpcpp_exec_ptr_ = new GKODpcppExecPtr(
            IOobject(device_exec_store, db),
            gko::give(gko::DpcppExecutor::create(0, omp_exec_ptr_->get_ptr())));
    }
    if (device_executor_name_ == "hip") {
        hip_exec_ptr_ =
            new GKOHipExecPtr(IOobject(device_exec_store, db),
                              gko::give(gko::HipExecutor::create(
                                  0, omp_exec_ptr_->get_ptr(), true)));
    }
    if (device_executor_name_ == "omp") {
        omp_exec_ptr_ = new GKOOmpExecPtr(IOobject(device_exec_store, db),
                                          omp_exec_ptr_->get_ptr());
    }
}

defineTemplateTypeNameWithName(GKOExecPtr, "ExecIOPtr");
defineTemplateTypeNameWithName(GKOCudaExecPtr, "CudaExecIOPtr");
defineTemplateTypeNameWithName(GKOOmpExecPtr, "OmpExecIOPtr");
defineTemplateTypeNameWithName(GKOHipExecPtr, "HipExecIOPtr");
defineTemplateTypeNameWithName(GKOReferenceExecPtr, "ReferenceExecIOPtr");

}  // namespace Foam
