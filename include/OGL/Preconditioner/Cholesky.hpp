// SPDX-FileCopyrightText: 2025 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "OGL/Preconditioner/Schwarz.hpp"

class Cholesky  // : public PreconditionerWrapper
{
    using dic = gko::preconditioner::Jacobi<double, label>;
    using fic = gko::preconditioner::Jacobi<float, label>;

    std::shared_ptr<gko::Executor> exec_;
    std::shared_ptr<const gko::LinOp> mtx_;
    const dictionary &d_;
    const label verbose_;
    bool skip_sorting_;
    bool multi_level_schwarz_;
    word precision_;
    word factorization_;


public:
    Cholesky(std::shared_ptr<gko::Executor> exec,
             std::shared_ptr<const gko::LinOp> mtx, const dictionary &d,
             label verbose)
        : exec_(exec),
          mtx_(mtx),
          d_(d),
          verbose_(verbose),
          skip_sorting_(d.lookupOrDefault<Switch>("skipSorting", true)),
          multi_level_schwarz_(
              d.lookupOrDefault<Switch>("multiLevelSchwarz", false)),
          precision_(d.lookupOrDefault("precision", word("double"))),
          factorization_(d.lookupOrDefault("factorization", word("IC")))
    {
        word msg = "Generate " + factorization_ +
                   "  preconditioner:\n\tprecision: " + precision_;
        MLOG_0(verbose_, msg)
    }


    std::shared_ptr<gko::LinOp> generate_factorization() const
    {
        if (factorization_ == "IC") {
            auto factorization_factory =
                gko::factorization::Ic<scalar, label>::build()
                    .with_skip_sorting(skip_sorting_)
                    .on(exec_);

            auto gkodistmatrix =
                gko::as<RepartDistMatrix>(mtx_)->get_dist_matrix();
            return factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<scalar, label,
                                                               label>>(
                    gkodistmatrix)
                    ->get_local_matrix());
        }
        if (factorization_ == "ParIC") {
            label iterations = d_.lookupOrDefault("iterations", label(5));
            auto factorization_factory =
                gko::factorization::ParIct<scalar, label>::build()
                    .with_skip_sorting(skip_sorting_)
                    .with_iterations(iterations)
                    .on(exec_);

            auto gkodistmatrix =
                gko::as<RepartDistMatrix>(mtx_)->get_dist_matrix();
            return factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<scalar, label,
                                                               label>>(
                    gkodistmatrix)
                    ->get_local_matrix());
        }
        if (factorization_ == "ParICT") {
            label iterations = d_.lookupOrDefault("iterations", label(5));
            label fillInLimit = d_.lookupOrDefault("fillInLimit", label(2));
            auto factorization_factory =
                gko::factorization::ParIct<scalar, label>::build()
                    .with_skip_sorting(skip_sorting_)
                    .with_fill_in_limit(fillInLimit)
                    .with_iterations(iterations)
                    .on(exec_);

            auto gkodistmatrix =
                gko::as<RepartDistMatrix>(mtx_)->get_dist_matrix();
            return factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<scalar, label,
                                                               label>>(
                    gkodistmatrix)
                    ->get_local_matrix());
        }
    }

    virtual std::shared_ptr<gko::LinOp> create()
    {
        auto precond_factory = gko::preconditioner::Ic<>::build().on(exec_);
        return dispatch_schwarz(
            mtx_, exec_,
            gko::share(precond_factory->generate(generate_factorization())), d_,
            verbose_);
    }
};
