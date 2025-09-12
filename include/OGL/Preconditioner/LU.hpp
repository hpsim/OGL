// SPDX-FileCopyrightText: 2025 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "OGL/Preconditioner/Schwarz.hpp"

class LU  // : public PreconditionerWrapper
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

public:
    LU(std::shared_ptr<gko::Executor> exec,
                std::shared_ptr<const gko::LinOp> mtx, const dictionary &d,
                label verbose)
        : exec_(exec),
          mtx_(mtx),
          d_(d),
          verbose_(verbose),
          skip_sorting_(d.lookupOrDefault<Switch>("skipSorting", true)),
          multi_level_schwarz_(
              d.lookupOrDefault<Switch>("multiLevelSchwarz", false)),
          precision_(d.lookupOrDefault("precision", word("double")))
    {
        word msg = "Generate incomplete LU (ILU) preconditioner:\n\tprecision: " +
                   precision_; 
        MLOG_0(verbose_, msg)
    }

    virtual std::shared_ptr<gko::LinOp> create()
    {
            label iterations {10};//(d.lookupOrDefault("iterations", label(1)));

	auto precond_factory = gko::share(
		gko::preconditioner::Ilu<>::build().on(exec_));

        auto wrapper = [this](auto f) {
            auto factorization_factory =
                gko::factorization::Ilu<scalar, label>::build()
                    .with_skip_sorting(skip_sorting_)
                    //.with_iterations(iterations)
                    .on(exec_);
            auto gkodistmatrix =
                gko::as<RepartDistMatrix>(mtx_)->get_dist_matrix();
            auto factorization = gko::share(factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<
                    scalar, label, label>>(gkodistmatrix)
                    ->get_local_matrix()));
            if (multi_level_schwarz_) {
                // auto distmtx =
                //     gko::as<RepartDistMatrix>(mtx_)->get_dist_matrix();
                // auto local = gko::as<RepartDistMatrix>(mtx_)->get_local();
                // auto local_rows = local->get_size()[0];
                // return wrap_multi_level_schwarz(distmtx, exec_, f, d_,
                //                                 local_rows, verbose_);
            } else {
                return wrap_schwarz(mtx_, exec_, std::move(f));
            return wrap_schwarz(gkodistmatrix, exec_,
                                std::move(f), factorization);
            }
        };

	return wrapper(precond_factory);

    }
};
