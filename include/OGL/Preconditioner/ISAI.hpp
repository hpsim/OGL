// SPDX-FileCopyrightText: 2025 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "OGL/Preconditioner/Schwarz.hpp"

class ISAI  // : public PreconditionerWrapper
{
    using dic = gko::preconditioner::Jacobi<double, label>;
    using fic = gko::preconditioner::Jacobi<float, label>;

    std::shared_ptr<gko::Executor> exec_;
    std::shared_ptr<const gko::LinOp> mtx_;
    const dictionary &d_;
    const label verbose_;
    bool skip_sorting_;
    bool multi_level_schwarz_;
    word type_;
    label sparsityPower_;


public:
    ISAI(std::shared_ptr<gko::Executor> exec,
         std::shared_ptr<const gko::LinOp> mtx, const dictionary &d,
         label verbose)
        : exec_(exec),
          mtx_(mtx),
          d_(d),
          verbose_(verbose),
          skip_sorting_(d.lookupOrDefault<Switch>("skipSorting", true)),
          multi_level_schwarz_(
              d.lookupOrDefault<Switch>("multiLevelSchwarz", false)),
          type_(d.lookupOrDefault("type", word("general"))),
          sparsityPower_(d.lookupOrDefault("sparsityPower", label(1)))
    {
        word msg = "Generate " + type_ + "ISAI" +
                   "  preconditioner:\n\tsparsityPower: " +
                   std::to_string(sparsityPower_);
        MLOG_0(verbose_, msg)
    }


    auto generate_precond_factory()
    {
        if (type_ == "SPD") {
            return gko::preconditioner::Isai<
                       gko::preconditioner::isai_type::spd, scalar,
                       label>::build()
                .with_skip_sorting(skip_sorting_)
                .with_sparsity_power(sparsityPower_)
                .on(exec_);
        }
        if (type_ == "General") {
            return gko::preconditioner::Isai<
                       gko::preconditioner::isai_type::general, scalar,
                       label>::build()
                .with_skip_sorting(skip_sorting_)
                .with_sparsity_power(sparsityPower_)
                .on(exec_);
        }
    }

    virtual std::shared_ptr<gko::LinOp> create()
    {
        auto wrapper = [this](auto f) {
            if (multi_level_schwarz_) {
                // auto distmtx =
                //     gko::as<RepartDistMatrix>(mtx_)->get_dist_matrix();
                // auto local = gko::as<RepartDistMatrix>(mtx_)->get_local();
                // auto local_rows = local->get_size()[0];
                // return wrap_multi_level_schwarz(distmtx, exec_, f, d_,
                //                                 local_rows, verbose_);
            } else {
                return wrap_schwarz(mtx_, exec_, std::move(f));
            }
        };

        return wrapper(generate_precond_factory());
    }
};
