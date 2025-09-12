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
          precision_(d.lookupOrDefault("precision", word("double")))
    {
        word msg = "Generate incomplete cholesky (IC) preconditioner:\n\tprecision: " +
                   precision_; 
        MLOG_0(verbose_, msg)
    }

    virtual std::shared_ptr<gko::LinOp> create()
    {
     auto factorization_factory = gko::share(
	gko::factorization::Ic<scalar, label>::build()
	    .with_skip_sorting(skip_sorting_)
	    .on(exec_));
     auto gkodistmatrix =
	 gko::as<RepartDistMatrix>(mtx_)->get_dist_matrix();
     auto factorization =  gko::share(factorization_factory->generate(
	 gko::as<gko::experimental::distributed::Matrix<
	     scalar, label, label>>(gkodistmatrix)
	     ->get_local_matrix()));
     auto precond_factory = gko::share(
	 gko::preconditioner::Ic<>::build().on(exec_));
	
    return wrap_schwarz(gkodistmatrix, exec_,
			std::move(precond_factory), factorization);

        // auto wrapper = [this](auto f) {
        //     if (multi_level_schwarz_) {
        //         auto distmtx =
        //             gko::as<RepartDistMatrix>(mtx_)->get_dist_matrix();
        //         auto local = gko::as<RepartDistMatrix>(mtx_)->get_local();
        //         auto local_rows = local->get_size()[0];
        //         return wrap_multi_level_schwarz(distmtx, exec_, f, d_,
        //                                         local_rows, verbose_);
        //     } else {
        //         return wrap_schwarz(mtx_, exec_, std::move(f));
        //     }
        // };
        //
        // return wrapper(factorization_factory);
    }
};
