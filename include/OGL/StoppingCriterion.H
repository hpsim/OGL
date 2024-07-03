// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "OGL/common.H"

#include "fvCFD.H"

namespace Foam {


class StoppingCriterion {
    using vec = gko::matrix::Dense<scalar>;
    using mtx = gko::matrix::Csr<scalar>;
    using val_array = gko::array<scalar>;
    using idx_array = gko::array<label>;

    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    class OpenFOAMDistStoppingCriterion
        : public gko::EnablePolymorphicObject<OpenFOAMDistStoppingCriterion,
                                              gko::stop::Criterion> {
        friend class gko::EnablePolymorphicObject<OpenFOAMDistStoppingCriterion,
                                                  gko::stop::Criterion>;
        using Criterion = gko::stop::Criterion;

    public:
        GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
        {
            /**
             * Boolean set by the user to stop the iteration process
             */
            // TODO check why GKO_FACTORY_PARAMETER_SCALAR does not work
            scalar GKO_FACTORY_PARAMETER(openfoam_absolute_tolerance, 1.0e-6);

            scalar GKO_FACTORY_PARAMETER(openfoam_relative_tolerance, 0.0);

            label GKO_FACTORY_PARAMETER(openfoam_minIter, 0);

            label GKO_FACTORY_PARAMETER(openfoam_maxIter, 0);

            label GKO_FACTORY_PARAMETER(frequency, 1);

            std::add_pointer<label>::type GKO_FACTORY_PARAMETER_SCALAR(iter,
                                                                       NULL);

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(time,
                                                                        NULL);

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(
                residual_norm, NULL);

            std::shared_ptr<vec> GKO_FACTORY_PARAMETER_SCALAR(residual_norms,
                                                              {});

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(
                init_residual_norm, NULL);

            label GKO_FACTORY_PARAMETER(verbose, 0);

            bool GKO_FACTORY_PARAMETER(export_res, false);

            std::shared_ptr<gko::LinOp> GKO_FACTORY_PARAMETER(gkomatrix, {});

            std::shared_ptr<dist_vec> GKO_FACTORY_PARAMETER(x, {});

            std::shared_ptr<dist_vec> GKO_FACTORY_PARAMETER(b, {});
        };

        GKO_ENABLE_CRITERION_FACTORY(OpenFOAMDistStoppingCriterion, parameters,
                                     Factory);

        GKO_ENABLE_BUILD_METHOD(Factory);

        /* Compute the SpMV of A with x_ref, where x_ref is a vector containing
         * the average of x in every row. This is needed to initialise the
         * normfactor in the first iteration.
         *  */
        void compute_Axref_dist(
            size_t global_size, size_t local_size,
            std::shared_ptr<const gko::Executor> device_exec,
            std::shared_ptr<const gko::LinOp> gkomatrix,
            std::shared_ptr<const dist_vec> x,
            std::shared_ptr<dist_vec> res) const;

        /* Compute the normfactor ie || Ax - x* || + || b - x* ||
         * or rewritten as || r - ( b - x* ) || + || (b - x*) ||
         *  */
        scalar compute_normfactor_dist(
            std::shared_ptr<const gko::Executor> device_exec, const dist_vec *r,
            std::shared_ptr<const gko::LinOp> gkomatrix,
            std::shared_ptr<const dist_vec> x,
            std::shared_ptr<const dist_vec> b) const;

        /* Implementation of the residual norm check
         *  */
        bool check_impl(gko::uint8 stoppingId, bool setFinalized,
                        gko::array<gko::stopping_status> *stop_status,
                        bool *one_changed,
                        const Criterion::Updater &updater) override;


        explicit OpenFOAMDistStoppingCriterion(
            std::shared_ptr<const gko::Executor> exec)
            : EnablePolymorphicObject<OpenFOAMDistStoppingCriterion, Criterion>(
                  std::move(exec))
        {}

        explicit OpenFOAMDistStoppingCriterion(const Factory *factory,
                                               const gko::stop::CriterionArgs &)

            : EnablePolymorphicObject<OpenFOAMDistStoppingCriterion, Criterion>(
                  factory->get_executor()),
              parameters_{factory->get_parameters()}
        {}

        void set_eval_norm_factor(bool eval_norm_factor)
        {
            eval_norm_factor_ = eval_norm_factor;
        }

        mutable bool first_iter_ = true;

        mutable scalar norm_factor_ = 1;

        mutable bool eval_norm_factor_ = true;

        mutable std::vector<scalar> res_norms_{};
    };

    mutable label maxIter_;

    const label minIter_;

    const scalar tolerance_;

    const scalar relTol_;

    const scalar res_norm_eval_;

    const label norm_eval_limit_;

    const label frequency_;

    const scalar relaxationFactor_;

    const bool adapt_minIter_;

    const std::shared_ptr<vec> normalised_res_norms_;

    mutable scalar init_normalised_res_norm_;

    mutable scalar normalised_res_norm_;

    mutable label iter_;

    mutable scalar time_;

public:
    StoppingCriterion(const dictionary &controlDict)
        : maxIter_(controlDict.lookupOrDefault("maxIter", label(1000))),
          minIter_(controlDict.lookupOrDefault("minIter", label(0))),
          tolerance_(controlDict.lookupOrDefault("tolerance", scalar(1e-6))),
          relTol_(controlDict.lookupOrDefault("relTol", scalar(1e-6))),
          res_norm_eval_(
              controlDict.lookupOrDefault("resNormEval", scalar(0.1))),
          norm_eval_limit_(
              controlDict.lookupOrDefault("normEvalLimit", label(100))),
          frequency_(controlDict.lookupOrDefault("evalFrequency", label(1))),
          relaxationFactor_(
              controlDict.lookupOrDefault("relaxationFactor", scalar(0.6))),
          adapt_minIter_(
              controlDict.lookupOrDefault<Switch>("adaptMinIter", true)),
          normalised_res_norms_(gko::share(vec::create(
              gko::ReferenceExecutor::create(),
              gko::dim<2>{static_cast<gko::dim<2>::dimension_type>(maxIter_),
                          1}))),
          init_normalised_res_norm_(0),
          normalised_res_norm_(0),
          iter_(0),
          time_(0)
    {
        normalised_res_norms_->fill(0.0);
        if (word(controlDict.lookup("solver")) == "GKOBiCGStab") maxIter_ *= 2;
    }

    std::shared_ptr<const gko::stop::CriterionFactory>
    build_dist_stopping_criterion(std::shared_ptr<gko::Executor> device_exec,
                                  std::shared_ptr<gko::LinOp> gkomatrix,
                                  std::shared_ptr<dist_vec> x,
                                  std::shared_ptr<dist_vec> b, label verbose,
                                  bool export_res, label prev_solve_iters,
                                  scalar prev_rel_cost) const
    {
        label minIter = minIter_;
        label frequency = frequency_;
        if (!export_res) {
            if (prev_solve_iters > 0 && adapt_minIter_ && prev_rel_cost > 0) {
                minIter = prev_solve_iters * relaxationFactor_;
                auto alpha =
                    sqrt(1.0 / (prev_solve_iters * (1.0 - relaxationFactor_)) *
                         prev_rel_cost);
                frequency = min(norm_eval_limit_, max(1, label(1 / alpha)));
            }
        }

        word msg = "Creating stopping criterion with minIter " +
                   std::to_string(minIter) + " frequency " +
                   std::to_string(frequency);

        MLOG_0(verbose, msg)

        return OpenFOAMDistStoppingCriterion::build()
            .with_openfoam_absolute_tolerance(tolerance_)
            .with_openfoam_relative_tolerance(relTol_)
            .with_openfoam_minIter(minIter)
            .with_openfoam_maxIter(maxIter_)
            .with_frequency(frequency)
            .with_verbose(verbose)
            .with_export_res(export_res)
            .with_init_residual_norm(&init_normalised_res_norm_)
            .with_residual_norm(&normalised_res_norm_)
            .with_residual_norms(normalised_res_norms_)
            .with_iter(&iter_)
            .with_time(&time_)
            .with_gkomatrix(gkomatrix)
            .with_x(x)
            .with_b(b)
            .on(device_exec);
    }

    scalar get_init_res_norm() const { return init_normalised_res_norm_; }

    scalar get_res_norm() const { return normalised_res_norm_; }

    std::shared_ptr<vec> get_res_norms() const { return normalised_res_norms_; }

    label get_is_final() const { return relTol_ == 0.0; }

    label get_num_iters() const { return iter_; }

    scalar get_res_norm_time() const { return time_; }
};
}  // namespace Foam
