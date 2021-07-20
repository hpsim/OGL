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

#ifndef OGL_STOPPING_CRITERION_H
#define OGL_STOPPING_CRITERION_H

#include <ginkgo/ginkgo.hpp>
#include "fvCFD.H"

namespace Foam {

class StoppingCriterion {
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

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(
                residual_norm, NULL);

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(
                init_residual_norm, NULL);
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
            auto exec = dense_r->get_executor();
            auto res = vec::create(exec, gko::dim<2>{1, 1});
            auto vec_abs = gko::clone(dense_r);
            const label cells = dense_r->get_size()[0];
            vec_abs->compute_absolute_inplace();
            init_unit_vec_on_first_iter(dense_r, cells);

            unit_->compute_dot(gko::lend(vec_abs), gko::lend(res));

            auto res_host = vec::create(exec->get_master(), gko::dim<2>{1, 1});
            res_host->copy_from(gko::lend(res));

            bool result = false;
            scalar residual_norm =
                res_host->at(0) / parameters_.openfoam_norm_factor;

            *(parameters_.residual_norm) = residual_norm;

            // Store initial residual
            if (iter_ == 0) {
                *(parameters_.init_residual_norm) = residual_norm;
            }
            scalar init_residual = *(parameters_.init_residual_norm);

            // stop if maximum number of iterations was reached
            if (iter_ == parameters_.openfoam_maxIter) {
                result = true;
            } else if (iter_ > parameters_.openfoam_minIter) {
                // check if absolute tolerance is hit
                if (residual_norm < parameters_.openfoam_absolute_tolerance) {
                    result = true;
                }
                // check if relative tolerance is hit
                if (parameters_.openfoam_relative_tolerance > 0 &&
                    residual_norm < parameters_.openfoam_relative_tolerance *
                                        init_residual) {
                    result = true;
                }
            }

            if (result) {
                this->set_all_statuses(stoppingId, setFinalized, stop_status);
                *one_changed = true;
            }

            iter_++;
            return result;
        }

        void init_unit_vec_on_first_iter(const vec *dense_r, const size_t cells)
        {
            if (first_iter_) {
                unit_ = vec::create(dense_r->get_executor(),
                                    gko::dim<2>{(unsigned long)cells, 1});
                unit_->fill(1.0);
                first_iter_ = false;
            }
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

        mutable bool first_iter_ = true;

        mutable std::shared_ptr<vec> unit_ = {};
    };

    mutable label maxIter_;

    const label minIter_;

    const scalar tolerance_;

    const scalar relTol_;

    mutable scalar init_normalised_res_norm_;

    mutable scalar normalised_res_norm_;


public:
    StoppingCriterion(const dictionary &controlDict)
        : maxIter_(controlDict.lookupOrDefault("maxIter", label(1000))),
          minIter_(controlDict.lookupOrDefault("minIter", label(0))),
          tolerance_(controlDict.lookupOrDefault("tolerance", scalar(1e-6))),
          relTol_(controlDict.lookupOrDefault("relTol", scalar(1e-6))),
          init_normalised_res_norm_(0),
          normalised_res_norm_(0)
    {
        if (word(controlDict.lookup("solver")) == "GKOBiCGStab") maxIter_ *= 2;
    }

    std::shared_ptr<const gko::stop::CriterionFactory> build_stopping_criterion(
        std::shared_ptr<gko::Executor> device_exec, scalar norm_factor) const
    {
        return OpenFOAMStoppingCriterion::build()
            .with_openfoam_absolute_tolerance(tolerance_)
            .with_openfoam_relative_tolerance(relTol_)
            .with_openfoam_norm_factor(norm_factor)
            .with_openfoam_minIter(minIter_)
            .with_openfoam_maxIter(maxIter_)
            .with_init_residual_norm(&init_normalised_res_norm_)
            .with_residual_norm(&normalised_res_norm_)
            .on(device_exec);
    }

    scalar get_init_res_norm() const { return init_normalised_res_norm_; }

    scalar get_res_norm() const { return normalised_res_norm_; }
};
}  // namespace Foam

#endif
