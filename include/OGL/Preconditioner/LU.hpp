// SPDX-FileCopyrightText: 2025 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "OGL/Preconditioner/Schwarz.hpp"

class LU {
    std::shared_ptr<gko::Executor> exec_;
    std::shared_ptr<const gko::LinOp> mtx_;
    const dictionary &d_;
    const label verbose_;
    bool skip_sorting_;
    bool multi_level_schwarz_;
    word factorization_;
    label iterations_;
    label fillInLimit_;


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
          factorization_(d.lookupOrDefault("factorization", word("ParILU"))),
          iterations_(d.lookupOrDefault("iterations", label(5))),
          fillInLimit_(d.lookupOrDefault("fillInLimit", label(2)))
    {
        word msg = "\nGenerate ILU preconditioner:\n\tfactorization: " +
                   factorization_;
        if (factorization_ == "ParILU" || factorization_ == "ParILUT") {
            msg += "\n\titerations: " + std::to_string(iterations_);
        }
        if (factorization_ == "ParILUT") {
            msg += "\n\tfillInLimit: " + std::to_string(fillInLimit_);
        }
        msg += word("\n\tskipSorting: ") + Switch(skip_sorting_).c_str() +
               "\n\tmultiLevelSchwarz: " + Switch(multi_level_schwarz_).c_str();
        MLOG_0(verbose_, msg)
    }


    std::shared_ptr<const gko::LinOp> get_local_matrix() const
    {
        return gko::as<RepartDistMatrix>(mtx_)->get_local_matrix();
    }

    std::shared_ptr<gko::LinOp> generate_factorization() const
    {
        if (factorization_ == "ILU") {
            auto factorization_factory =
                gko::factorization::Ilu<scalar, label>::build()
                    .with_skip_sorting(skip_sorting_)
                    .on(exec_);

            return factorization_factory->generate(get_local_matrix());
        }
        if (factorization_ == "ParILU") {
            auto factorization_factory =
                gko::factorization::ParIlu<scalar, label>::build()
                    .with_skip_sorting(skip_sorting_)
                    .with_iterations(iterations_)
                    .on(exec_);

            return factorization_factory->generate(get_local_matrix());
        }
        if (factorization_ == "ParILUT") {
            auto factorization_factory =
                gko::factorization::ParIlut<scalar, label>::build()
                    .with_skip_sorting(skip_sorting_)
                    .with_fill_in_limit(fillInLimit_)
                    .with_iterations(iterations_)
                    .on(exec_);
            return factorization_factory->generate(get_local_matrix());
        }
        FatalErrorInFunction << "Unknown LU factorization: " << factorization_
                             << "\nValid Choices: ILU, ParILU, ParILUT"
                             << abort(FatalError);
        return {};
    }

    std::shared_ptr<gko::LinOp> create()
    {
        auto precond_factory = gko::preconditioner::Ilu<>::build().on(exec_);
        return dispatch_schwarz(
            mtx_, exec_,
            gko::share(precond_factory->generate(generate_factorization())), d_,
            verbose_);
    }
};
