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
    StoppingCriterion.H

\*---------------------------------------------------------------------------*/

#ifndef OGL_STOPPING_CRITERION_H
#define OGL_STOPPING_CRITERION_H

#include <ginkgo/ginkgo.hpp>
#include "../common/common.H"

#include "fvCFD.H"

namespace Foam {


class StoppingCriterion {
    using vec = gko::matrix::Dense<scalar>;
    using mtx = gko::matrix::Csr<scalar>;
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;

    using dist_vec = gko::distributed::Vector<scalar>;
    using dist_mtx = gko::distributed::Matrix<scalar, label>;

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

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(
                residual_norm, NULL);

            std::shared_ptr<vec> GKO_FACTORY_PARAMETER_SCALAR(residual_norms,
                                                              {});

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(
                init_residual_norm, NULL);

            label GKO_FACTORY_PARAMETER(verbose, 0);

            bool GKO_FACTORY_PARAMETER(export_res, false);

            std::shared_ptr<dist_mtx> GKO_FACTORY_PARAMETER(gkomatrix, {});

            std::shared_ptr<dist_vec> GKO_FACTORY_PARAMETER(x, {});

            std::shared_ptr<dist_vec> GKO_FACTORY_PARAMETER(b, {});
        };

        GKO_ENABLE_CRITERION_FACTORY(OpenFOAMDistStoppingCriterion, parameters,
                                     Factory);

        GKO_ENABLE_BUILD_METHOD(Factory);

    protected:
        /* Compute the SPMV of A with x_ref
         *
         *
         *  */
        void compute_Axref_dist(
            size_t global_size, size_t local_size,
            std::shared_ptr<const gko::Executor> device_exec,
            std::shared_ptr<const dist_mtx> gkomatrix,
            std::shared_ptr<dist_vec> x, std::shared_ptr<dist_vec> res) const
        {
            auto xAvg = gko::initialize<gko::matrix::Dense<scalar>>(
                1, {0}, device_exec);

            x->compute_average_unsafe(xAvg.get());

#ifdef GINKGO_WITH_OGL_EXTENSIONS
            gkomatrix->compute_column_vector_sum(res.get());
            res->scale(xAvg.get());
#else
            // if column vector sum is not availible use dot product
            auto xAvg_vec = gko::share(dist_vec::create(
                device_exec, x->get_communicator(), global_size, local_size));
            xAvg_vec->fill(1.0);
            xAvg_vec->scale(xAvg.get());

            // TODO needs a distributed res
            gkomatrix->apply(xAvg_vec.get(), res.get());
            // std::cout << "compute_Axref apply done" << endl;
#endif
        }


        scalar compute_normfactor_dist(
            std::shared_ptr<const gko::Executor> device_exec, const dist_vec *r,
            std::shared_ptr<const dist_mtx> gkomatrix,
            std::shared_ptr<dist_vec> x,
            std::shared_ptr<const dist_vec> b) const
        {
            // std::cout << "compute_normfactor dist" << endl;
            // SIMPLE_TIME(verbose_, compute_col_sum_A, [=]() {
            // computes        || Ax - x* ||        + || b - x* ||
            // or rewritten as || r - ( b - x* ) || + || (b - x*) ||
            const label nCells{(label)x->get_size()[0]};

            // TODO store colA vector

            auto comm = x->get_communicator();

            gko::dim<2> local_size = x->get_local()->get_size();

            // TODO FIXME
            gko::dim<2> global_size = x->get_local()->get_size();
            // gko::dim<2> global_size = x->get_size();

            auto Axref = gko::share(
                dist_vec::create(device_exec, comm, global_size, local_size));

            compute_Axref_dist(global_size[0], local_size[0], device_exec,
                               gkomatrix, x, Axref);

            // TODO use dist vect
            auto b_norm =
                vec::create(device_exec->get_master(), gko::dim<2>{1, 1});
            b->compute_norm1(b_norm.get());

            auto unity = gko::initialize<gko::matrix::Dense<scalar>>(
                1, {1.0}, device_exec);

            auto b_sub_xstar = b->clone();
            b_sub_xstar->sub_scaled(unity.get(), Axref.get());

            auto norm_part2 = b_sub_xstar->compute_absolute();

            b_sub_xstar->sub_scaled(unity.get(), r);
            b_sub_xstar->compute_absolute_inplace();

            b_sub_xstar->add_scaled(unity.get(), norm_part2.get());
            auto res = vec::create(device_exec, gko::dim<2>{1, 1});
            b_sub_xstar->compute_norm1(res.get());

            auto res_host =
                vec::create(device_exec->get_master(), gko::dim<2>{1, 1});
            res_host->copy_from(res.get());

            return res_host->get_values()[0];
        }

        bool check_impl(gko::uint8 stoppingId, bool setFinalized,
                        gko::Array<gko::stopping_status> *stop_status,
                        bool *one_changed,
                        const Criterion::Updater &updater) override
        {
            if (*(parameters_.iter) > 0 &&
                *(parameters_.iter) < parameters_.openfoam_minIter) {
                *(parameters_.iter) += 1;
                return false;
            }

            const auto exec = this->get_executor();

            if (*(parameters_.iter) % parameters_.frequency != 0) {
                *(parameters_.iter) += 1;
                return false;
            }

            // TODO move to free template function
            auto *dense_r = gko::as<dist_vec>(updater.residual_);
            auto norm1 = vec::create(exec, gko::dim<2>{1, 1});
            dense_r->compute_norm1(norm1.get());
            auto norm1_host =
                vec::create(exec->get_master(), gko::dim<2>{1, 1});
            norm1_host->copy_from(norm1.get());
            scalar residual_norm = norm1_host->at(0);
            // TODO end

            bool result = false;

            // Store initial residual
            if (*(parameters_.iter) == 0) {
                //
                if (eval_norm_factor_) {
                    norm_factor_ = compute_normfactor_dist(
                        exec, dense_r, parameters_.gkomatrix, parameters_.x,
                        parameters_.b);
                }

                *(parameters_.init_residual_norm) =
                    residual_norm / norm_factor_;
            }

            residual_norm /= norm_factor_;

            if (parameters_.export_res) {
                parameters_.residual_norms->at(*(parameters_.iter)) =
                    residual_norm;
            }

            *(parameters_.residual_norm) = residual_norm;

            scalar init_residual = *(parameters_.init_residual_norm);

            // stop if maximum number of iterations was reached
            if (*(parameters_.iter) == parameters_.openfoam_maxIter) {
                result = true;
            }
            // check if absolute tolerance is hit
            if (residual_norm < parameters_.openfoam_absolute_tolerance) {
                result = true;
            }
            // check if relative tolerance is hit
            if (parameters_.openfoam_relative_tolerance > 0 &&
                residual_norm <
                    parameters_.openfoam_relative_tolerance * init_residual) {
                result = true;
            }

            if (result) {
                this->set_all_statuses(stoppingId, setFinalized, stop_status);
                *one_changed = true;
            }

            *(parameters_.iter) += 1;
            return result;
        }


        explicit OpenFOAMDistStoppingCriterion(
            std::shared_ptr<const gko::Executor> exec)
            : EnablePolymorphicObject<OpenFOAMDistStoppingCriterion, Criterion>(
                  std::move(exec))
        {
            std::cout << " OpenFOAMDistStoppingCriterion " << endl;
        }

        explicit OpenFOAMDistStoppingCriterion(
            const Factory *factory, const gko::stop::CriterionArgs &args)

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

    class OpenFOAMStoppingCriterion
        : public gko::EnablePolymorphicObject<OpenFOAMStoppingCriterion,
                                              gko::stop::Criterion> {
        friend class gko::EnablePolymorphicObject<OpenFOAMStoppingCriterion,
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

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(
                residual_norm, NULL);

            std::shared_ptr<vec> GKO_FACTORY_PARAMETER_SCALAR(residual_norms,
                                                              {});

            std::add_pointer<scalar>::type GKO_FACTORY_PARAMETER_SCALAR(
                init_residual_norm, NULL);

            label GKO_FACTORY_PARAMETER(verbose, 0);

            bool GKO_FACTORY_PARAMETER(export_res, false);

            std::shared_ptr<mtx> GKO_FACTORY_PARAMETER(gkomatrix, {});

            std::shared_ptr<vec> GKO_FACTORY_PARAMETER(x, {});

            std::shared_ptr<vec> GKO_FACTORY_PARAMETER(b, {});
        };

        GKO_ENABLE_CRITERION_FACTORY(OpenFOAMStoppingCriterion, parameters,
                                     Factory);

        GKO_ENABLE_BUILD_METHOD(Factory);

    protected:
        void compute_Axref(std::shared_ptr<const gko::Executor> device_exec,
                           std::shared_ptr<const mtx> gkomatrix,
                           std::shared_ptr<vec> x,
                           std::shared_ptr<vec> res) const
        {
            // auto xAvg = gko::initialize<gko::matrix::Dense<scalar>>(
            //     1, {0}, device_exec);

            auto xAvg = x->compute_average_unsafe();

#ifdef GINKGO_WITH_OGL_EXTENSIONS
            gkomatrix->compute_column_vector_sum(res.get());
            res->scale(xAvg.get());
#else
            // if column vector sum is not availible use dot product
            auto xAvg_vec = gko::share(vec::create(device_exec, x->get_size()));
            xAvg_vec->fill(1.0);
            xAvg_vec->scale(xAvg.get());

            gkomatrix->apply(xAvg_vec.get(), res.get());
#endif
        }


        scalar compute_normfactor(
            std::shared_ptr<const gko::Executor> device_exec, const vec *r,
            std::shared_ptr<const mtx> gkomatrix, std::shared_ptr<vec> x,
            std::shared_ptr<const vec> b) const
        {
            std::cout << "compute normfactor" << endl;
            // SIMPLE_TIME(verbose_, compute_col_sum_A, [=]() {
            // computes        || Ax - x* ||        + || b - x* ||
            // or rewritten as || r - ( b - x* ) || + || (b - x*) ||
            const label nCells{(label)x->get_size()[0]};

            // TODO store colA vector
            // TODO move out of function?
            auto Axref =
                gko::share(vec::create(device_exec, gko::dim<2>{nCells, 1}));

            compute_Axref(device_exec, gkomatrix, x, Axref);

            auto b_norm =
                vec::create(device_exec->get_master(), gko::dim<2>{1, 1});
            b->compute_norm1(b_norm.get());

            auto unity = gko::initialize<gko::matrix::Dense<scalar>>(
                1, {1.0}, device_exec);

            auto b_sub_xstar = b->clone();
            b_sub_xstar->sub_scaled(unity.get(), Axref.get());

            auto norm_part2 = b_sub_xstar->compute_absolute();

            b_sub_xstar->sub_scaled(unity.get(), r);
            b_sub_xstar->compute_absolute_inplace();

            b_sub_xstar->add_scaled(unity.get(), norm_part2.get());
            auto res = vec::create(device_exec, gko::dim<2>{1, 1});
            b_sub_xstar->compute_norm1(res.get());

            auto res_host =
                vec::create(device_exec->get_master(), gko::dim<2>{1, 1});
            res_host->copy_from(res.get());

            return res_host->get_values()[0];
        }

        bool check_impl(gko::uint8 stoppingId, bool setFinalized,
                        gko::Array<gko::stopping_status> *stop_status,
                        bool *one_changed,
                        const Criterion::Updater &updater) override
        {
            std::cout << "check_impl serial" << endl;
            if (*(parameters_.iter) > 0 &&
                *(parameters_.iter) < parameters_.openfoam_minIter) {
                *(parameters_.iter) += 1;
                return false;
            }

            const auto exec = this->get_executor();

            if (*(parameters_.iter) % parameters_.frequency != 0) {
                *(parameters_.iter) += 1;
                return false;
            }

            auto *dense_r = gko::as<vec>(updater.residual_);
            auto norm1 = vec::create(exec, gko::dim<2>{1, 1});
            dense_r->compute_norm1(norm1.get());
            auto norm1_host =
                vec::create(exec->get_master(), gko::dim<2>{1, 1});
            norm1_host->copy_from(norm1.get());
            scalar residual_norm = norm1_host->at(0);

            bool result = false;

            // Store initial residual
            if (*(parameters_.iter) == 0) {
                //
                if (eval_norm_factor_) {
                    norm_factor_ =
                        compute_normfactor(exec, dense_r, parameters_.gkomatrix,
                                           parameters_.x, parameters_.b);
                }

                *(parameters_.init_residual_norm) =
                    residual_norm / norm_factor_;
            }

            residual_norm /= norm_factor_;

            if (parameters_.export_res) {
                parameters_.residual_norms->at(*(parameters_.iter)) =
                    residual_norm;
            }

            *(parameters_.residual_norm) = residual_norm;

            scalar init_residual = *(parameters_.init_residual_norm);

            // stop if maximum number of iterations was reached
            if (*(parameters_.iter) == parameters_.openfoam_maxIter) {
                result = true;
            }
            // check if absolute tolerance is hit
            if (residual_norm < parameters_.openfoam_absolute_tolerance) {
                result = true;
            }
            // check if relative tolerance is hit
            if (parameters_.openfoam_relative_tolerance > 0 &&
                residual_norm <
                    parameters_.openfoam_relative_tolerance * init_residual) {
                result = true;
            }

            if (result) {
                this->set_all_statuses(stoppingId, setFinalized, stop_status);
                *one_changed = true;
            }

            *(parameters_.iter) += 1;
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

    const label frequency_;

    const scalar relaxationFactor_;

    const bool adapt_minIter_;

    const std::shared_ptr<vec> normalised_res_norms_;

    mutable scalar init_normalised_res_norm_;

    mutable scalar normalised_res_norm_;

    mutable label iter_;


public:
    StoppingCriterion(const dictionary &controlDict)
        : maxIter_(controlDict.lookupOrDefault("maxIter", label(1000))),
          minIter_(controlDict.lookupOrDefault("minIter", label(0))),
          tolerance_(controlDict.lookupOrDefault("tolerance", scalar(1e-6))),
          relTol_(controlDict.lookupOrDefault("relTol", scalar(1e-6))),
          frequency_(controlDict.lookupOrDefault("evalFrequency", label(1))),
          relaxationFactor_(
              controlDict.lookupOrDefault("relaxationFactor", scalar(0.8))),
          adapt_minIter_(
              controlDict.lookupOrDefault<Switch>("adaptMinIter", true)),
          normalised_res_norms_(gko::share(vec::create(
              gko::ReferenceExecutor::create(), gko::dim<2>{maxIter_, 1}))),
          init_normalised_res_norm_(0),
          normalised_res_norm_(0),
          iter_(0)
    {
        normalised_res_norms_->fill(0.0);
        if (word(controlDict.lookup("solver")) == "GKOBiCGStab") maxIter_ *= 2;
    }

    std::shared_ptr<const gko::stop::CriterionFactory> build_stopping_criterion(
        std::shared_ptr<gko::Executor> device_exec,
        std::shared_ptr<mtx> gkomatrix, std::shared_ptr<vec> x,
        std::shared_ptr<vec> b, label verbose, bool export_res,
        label prev_solve_iters) const
    {
        label minIter = minIter_;
        label frequency = frequency_;
        if (!export_res) {
            if (prev_solve_iters > 0 && adapt_minIter_) {
                minIter = prev_solve_iters * relaxationFactor_;
                frequency = max(1, minIter / 20);
            }
        }

        word msg = "Creating stopping criterion with minIter " +
                   std::to_string(minIter) + " frequency " +
                   std::to_string(frequency);

        LOG_1(verbose, msg)

        return OpenFOAMStoppingCriterion::build()
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
            .with_gkomatrix(gkomatrix)
            .with_x(x)
            .with_b(b)
            .on(device_exec);
    }

    std::shared_ptr<const gko::stop::CriterionFactory>
    build_dist_stopping_criterion(std::shared_ptr<gko::Executor> device_exec,
                                  std::shared_ptr<dist_mtx> gkomatrix,
                                  std::shared_ptr<dist_vec> x,
                                  std::shared_ptr<dist_vec> b, label verbose,
                                  bool export_res, label prev_solve_iters) const
    {
        std::cout << "build dist_stopping_criterion " << endl;
        label minIter = minIter_;
        label frequency = frequency_;
        if (!export_res) {
            if (prev_solve_iters > 0 && adapt_minIter_) {
                minIter = prev_solve_iters * relaxationFactor_;
                frequency = max(1, minIter / 20);
            }
        }

        word msg = "Creating stopping criterion with minIter " +
                   std::to_string(minIter) + " frequency " +
                   std::to_string(frequency);

        LOG_1(verbose, msg)

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
            .with_gkomatrix(gkomatrix)
            .with_x(x)
            .with_b(b)
            .on(device_exec);
    }

    scalar get_init_res_norm() const { return init_normalised_res_norm_; }

    scalar get_res_norm() const { return normalised_res_norm_; }

    std::shared_ptr<vec> get_res_norms() const
    {
        // std::unique_ptr<val_array> norms_array{new val_array{
        //     exec, normalised_res_norms_.begin(),
        //     normalised_res_norms_.end()}};

        // auto res_norms = gko::share(
        //     vec::create(exec->get_master(), gko::dim<2>(res_norms_.size(),
        //     1),
        //                 norms_array->as_view(), 1));

        // std::string fn_x{"export/p_res_norms.mtx"};
        // std::ofstream stream_x{fn_x};
        // std::cerr << "Writing " << fn_x << std::endl;
        // gko::write(stream_x, res_norms.get());

        return normalised_res_norms_;
    }

    label get_num_iters() const { return iter_; }
};
}  // namespace Foam

#endif
