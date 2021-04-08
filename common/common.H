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
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/
#ifndef OGL_COMMON_H
#define OGL_COMMON_H

#include "LduMatrix.H"
#include "fvCFD.H"
#include "regIOobject.H"

#include <ginkgo/ginkgo.hpp>
#include <map>

#define SIMPLE_TIME(NAME, F)                                            \
    auto start_##NAME = std::chrono::steady_clock::now();               \
    F auto end_##NAME = std::chrono::steady_clock::now();               \
    std::cout << "Gingko " #NAME " : "                                  \
              << std::chrono::duration_cast<std::chrono::milliseconds>( \
                     end_##NAME - start_##NAME)                         \
                     .count()                                           \
              << " ms\n";

namespace Foam {

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

typedef IOPtr<gko::matrix::Coo<scalar>> GKOCOOIOPtr;
typedef IOPtr<gko::Executor> GKOExecPtr;
typedef IOPtr<gko::CudaExecutor> GKOCudaExecPtr;
typedef IOPtr<gko::ReferenceExecutor> GKOReferenceExecPtr;
typedef IOPtr<gko::OmpExecutor> GKOOmpExecPtr;
typedef IOPtr<gko::HipExecutor> GKOHipExecPtr;

template <>
inline bool typeGlobal<GKOCOOIOPtr>()
{
    return true;
}
template <>
inline bool typeGlobal<GKOExecPtr>()
{
    return true;
}

class OpenFOAMStoppingCriterion
    : public gko::EnablePolymorphicObject<OpenFOAMStoppingCriterion,
                                          gko::stop::Criterion> {
    friend class gko::EnablePolymorphicObject<OpenFOAMStoppingCriterion,
                                              gko::stop::Criterion>;
    using Criterion = gko::stop::Criterion;
    using vec = gko::matrix::Dense<scalar>;

private:
    label iter_ = 0;

public:
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Boolean set by the user to stop the iteration process
         */
        // TODO check why GKO_FACTORY_PARAMETER_SCALAR does not work
        scalar GKO_FACTORY_PARAMETER(openfoam_absolute_tolerance, 1.0e-6);

        scalar GKO_FACTORY_PARAMETER(openfoam_relative_tolerance, 0.0);

        scalar GKO_FACTORY_PARAMETER(openfoam_norm_factor, 1.0);

        label GKO_FACTORY_PARAMETER(openfoam_minIter, 0);

        label GKO_FACTORY_PARAMETER(openfoam_maxIter, 0);
    };

    GKO_ENABLE_CRITERION_FACTORY(OpenFOAMStoppingCriterion, parameters,
                                 Factory);

    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    bool check_impl(gko::uint8 stoppingId, bool setFinalized,
                    gko::Array<gko::stopping_status> *stop_status,
                    bool *one_changed,
                    const Criterion::Updater &updater) override
    {
        iter_++;

        auto *dense_r = gko::as<vec>(updater.residual_);
        auto exec = dense_r->get_executor();
        auto res = vec::create(exec, gko::dim<2>{1, 1});
        auto vec_abs = gko::clone(dense_r);
        const label cells = dense_r->get_size()[0];
        vec_abs->compute_absolute_inplace();
        auto unit = vec::create(dense_r->get_executor(), gko::dim<2>{cells, 1});
        unit->fill(1.0);

        const auto unit_t = unit->transpose();

        unit_t->apply(gko::lend(vec_abs), gko::lend(res));

        auto res_host = vec::create(exec->get_master(), gko::dim<2>{1, 1});
        res_host->copy_from(gko::lend(res));

        bool result = false;
        scalar normalised_res_norm =
            res_host->at(0) / parameters_.openfoam_norm_factor;
        // printf("normalised res norm %f\n", normalised_res_norm);
        // printf("iter %d\n", iter_);

        if ((normalised_res_norm < parameters_.openfoam_absolute_tolerance) &&
            (iter_ >= parameters_.openfoam_minIter)) {
            result = true;
        }

        if (iter_ >= parameters_.openfoam_maxIter) {
            result = true;
        }

        if (result) {
            this->set_all_statuses(stoppingId, setFinalized, stop_status);
            *one_changed = true;
        }

        return result;
    }

    explicit OpenFOAMStoppingCriterion(
        std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<OpenFOAMStoppingCriterion, Criterion>(
              std::move(exec))
    {}

    explicit OpenFOAMStoppingCriterion(const Factory *factory,
                                       const gko::stop::CriterionArgs &args)

        : EnablePolymorphicObject<OpenFOAMStoppingCriterion, Criterion>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {}
};

class lduLduCommonBase {
    const objectRegistry &db_;

    const label maxIter_;

    const label minIter_;

    const scalar tolerance_;

    const scalar relTol_;

    const word update_sysMatrix_;

    const label nCells_;

    const label nNeighbours_;

    const label nElems_;

    const bool export_;

    // flag to set wheter the idxs for creating the gko coo matrix
    // should be sorted
    const bool sort_;

    const word device_executor_name_;

    const word app_executor_name_;

    // if sorting_idxs_ was found in object registry it does not
    // need to be resorted
    mutable bool is_sorted_;

    // executor where Ginkgo will perform the computation
    GKOExecPtr *device_exec_ptr_;

    // executor of the application
    GKOExecPtr *app_exec_ptr_;

    GKOReferenceExecPtr *ref_exec_ptr_;

    GKOCudaExecPtr *cuda_exec_ptr_;

    GKOOmpExecPtr *omp_exec_ptr_;

    GKOHipExecPtr *hip_exec_ptr_;

public:
    lduLduCommonBase(const objectRegistry &db, label nCells, label nNeighbours,
                     const dictionary &controlDict)
        : db_(db),
          maxIter_(controlDict.lookupOrDefault("maxIter", label(1000))),
          minIter_(controlDict.lookupOrDefault("minIter", label(0))),
          tolerance_(controlDict.lookupOrDefault("tolerance", scalar(1e-6))),
          relTol_(controlDict.lookupOrDefault("relTol", scalar(1e-6))),
          update_sysMatrix_(
              controlDict.lookupOrDefault("updateSysMatrix", word("yes"))),
          nCells_(nCells),
          nNeighbours_(nNeighbours),
          nElems_(nCells_ + 2 * nNeighbours_),
          export_(controlDict.lookupOrDefault<bool>("export", false)),
          sort_(controlDict.lookupOrDefault<bool>("sort", true)),
          device_executor_name_(
              controlDict.lookupOrDefault("executor", word("reference"))),
          app_executor_name_(
              controlDict.lookupOrDefault("app_executor", word("reference")))
    {
        // create executors
        bool app_exec_stored =
            get_db().foundObject<regIOobject>(app_executor_name_);

        if (app_exec_stored) {
            ref_exec_ptr_ = &get_db().lookupObjectRef<GKOReferenceExecPtr>(
                app_executor_name_);
            if (device_executor_name_ == app_executor_name_) {
                return;
            }
        }


        bool device_exec_stored =
            get_db().foundObject<regIOobject>(device_executor_name_);
        if (device_exec_stored) {
            if (device_executor_name_ == "omp") {
                omp_exec_ptr_ = &get_db().lookupObjectRef<GKOOmpExecPtr>(
                    device_executor_name_);
                return;
            }
            if (device_executor_name_ == "cuda") {
                cuda_exec_ptr_ = &get_db().lookupObjectRef<GKOCudaExecPtr>(
                    device_executor_name_);
                return;
            }
            if (device_executor_name_ == "hip") {
                hip_exec_ptr_ = &get_db().lookupObjectRef<GKOHipExecPtr>(
                    device_executor_name_);
                return;
            }
        }

        const fileName app_exec_store = app_executor_name_;
        ref_exec_ptr_ = new GKOReferenceExecPtr(
            IOobject(app_exec_store, get_db()),
            gko::give(gko::ReferenceExecutor::create()));

        const fileName device_exec_store = device_executor_name_;
        const fileName omp_exec_store = "omp";
        omp_exec_ptr_ = new GKOOmpExecPtr(IOobject(omp_exec_store, get_db()),
                                          gko::OmpExecutor::create());

        if (device_executor_name_ == "cuda") {
            cuda_exec_ptr_ =
                new GKOCudaExecPtr(IOobject(device_exec_store, get_db()),
                                   gko::give(gko::CudaExecutor::create(
                                       0, omp_exec_ptr_->get_ptr(), true)));
        }
        if (device_executor_name_ == "hip") {
            hip_exec_ptr_ =
                new GKOHipExecPtr(IOobject(device_exec_store, get_db()),
                                  gko::give(gko::HipExecutor::create(
                                      0, omp_exec_ptr_->get_ptr(), true)));
        }
        if (device_executor_name_ == "omp") {
            omp_exec_ptr_ =
                new GKOOmpExecPtr(IOobject(device_exec_store, get_db()),
                                  omp_exec_ptr_->get_ptr());
        }
    }

    const objectRegistry &get_db() { return db_; };

    scalar tolerance() const { return tolerance_; }

    label minIter() const { return minIter_; }

    label maxIter() const { return maxIter_; }

    label nCells() const { return nCells_; };

    label nElems() const { return nElems_; };

    label nNeighbours() const { return nNeighbours_; };

    //
    bool get_update_sys_matrix() const { return (update_sysMatrix_ == "yes"); }

    bool get_export() const { return export_; }

    bool get_sort() const { return sort_; }

    bool get_is_sorted() const { return is_sorted_; }

    void set_is_sorted(bool is_sorted) const { is_sorted_ = is_sorted; }

    word get_device_executor_name() const { return device_executor_name_; }

    std::shared_ptr<gko::Executor> exec() const
    {
        return device_exec_ptr_->get_ptr();
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
