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

#include <ginkgo/ginkgo.hpp>

#define SIMPLE_TIME(NAME, F)                                            \
    auto start_##NAME = std::chrono::steady_clock::now();               \
    F auto end_##NAME = std::chrono::steady_clock::now();               \
    std::cout << "Gingko " #NAME " : "                                  \
              << std::chrono::duration_cast<std::chrono::milliseconds>( \
                     end_##NAME - start_##NAME)                         \
                     .count()                                           \
              << " ms\n";

namespace Foam {

class OpenFOAMStoppingCriterion
    : public gko::EnablePolymorphicObject<OpenFOAMStoppingCriterion,
                                          gko::stop::Criterion> {
    friend class gko::EnablePolymorphicObject<OpenFOAMStoppingCriterion,
                                              gko::stop::Criterion>;
    using Criterion = gko::stop::Criterion;
    using vec = gko::matrix::Dense<scalar>;

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
        auto *dense_r = gko::as<vec>(updater.residual_);
        auto res = vec::create(dense_r->get_executor()->get_master(),
                               gko::dim<2>{1, 1});
        auto vec_abs = gko::clone(dense_r);
        const label cells = dense_r->get_size()[0];
        vec_abs->compute_absolute();
        auto unit = vec::create(dense_r->get_executor(), gko::dim<2>{cells, 1});
        unit->fill(1.0);

        const auto unit_t = unit->transpose();

        vec_abs->compute_dot(gko::lend(unit_t), gko::lend(res));

        printf("res %f, ", res->at(0));
        bool result = false;

        if (res->at(0) < parameters_.openfoam_absolute_tolerance) result = true;


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
    const label maxIter_;

    // TODO implement min iter
    const label minIter_;

    const scalar tolerance_;

    const scalar relTol_;

    label nCells_;

    label nNeighbours_;

    label nElems_;

    bool export_;

    // executor where Ginkgo will perform the computation
    std::shared_ptr<gko::Executor> exec_;

    // executor of the application
    std::shared_ptr<gko::Executor> app_exec_;

public:
    lduLduCommonBase(label maxIter, label minIter, scalar tolerance,
                     scalar relTolerance, label nCells, label nNeighbours,
                     const dictionary &controlDict)
        : maxIter_(maxIter),
          minIter_(minIter),
          tolerance_(tolerance),
          relTol_(relTolerance),
          nCells_(nCells),
          nNeighbours_(nNeighbours),
          export_(controlDict.lookupOrDefault<bool>("export", false))
    {
        // create executors
        auto executor_string =
            controlDict.lookupOrDefault("executor", word("reference"));
        auto app_executor_string =
            controlDict.lookupOrDefault("app_executor", word("reference"));

        const auto omp = gko::OmpExecutor::create();

        std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
            exec_map{
                {"omp", [&omp] { return omp; }},
                {"cuda",
                 [&omp] { return gko::CudaExecutor::create(0, omp, true); }},
                {"hip",
                 [&omp] { return gko::HipExecutor::create(0, omp, true); }},
                {"reference",
                 [&omp] { return gko::ReferenceExecutor::create(); }}};

        exec_ = exec_map.at(executor_string)();

        if (executor_string == app_executor_string) {
            app_exec_ = exec_;
        } else {
            app_exec_ = exec_map.at(app_executor_string)();
        }

        nElems_ = nCells_ + 2 * nNeighbours_;
    }

    scalar tolerance() const { return tolerance_; }

    label minIter() const { return minIter_; }

    label maxIter() const { return minIter_; }

    const label nCells() const { return nCells_; };

    const label nElems() const { return nElems_; };

    const label nNeighbours() const { return nNeighbours_; };

    bool get_export() const { return export_; }

    std::shared_ptr<gko::Executor> exec() const { return exec_; };

    std::shared_ptr<gko::Executor> app_exec() const { return app_exec_; };
};
}  // namespace Foam

#endif
