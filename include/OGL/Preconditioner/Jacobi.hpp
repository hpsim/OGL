// SPDX-FileCopyrightText: 2025 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "OGL/Preconditioner/Schwarz.hpp"

class BlockJacobi  // : public PreconditionerWrapper
{
    using dbj = gko::preconditioner::Jacobi<double, label>;
    using fbj = gko::preconditioner::Jacobi<float, label>;

    std::shared_ptr<gko::Executor> exec_;
    std::shared_ptr<const gko::LinOp> mtx_;
    const dictionary &d_;
    const label verbose_;
    bool skip_sorting_;
    bool multi_level_schwarz_;
    label max_block_size_;
    word precision_;


public:
    BlockJacobi(std::shared_ptr<gko::Executor> exec,
                std::shared_ptr<const gko::LinOp> mtx, const dictionary &d,
                label verbose)
        : exec_(exec),
          mtx_(mtx),
          d_(d),
          verbose_(verbose),
          skip_sorting_(d.lookupOrDefault<Switch>("skipSorting", true)),
          multi_level_schwarz_(
              d.lookupOrDefault<Switch>("multiLevelSchwarz", false)),
          max_block_size_(d.lookupOrDefault("maxBlockSize", label(1))),
          precision_(d.lookupOrDefault("precision", word("double")))
    {
        word msg = "Generate (Block)-Jacobi preconditioner:\n\tprecision: " +
                   precision_ + "\n\tmaxBlockSize " +
                   std::to_string(max_block_size_);
        MLOG_0(verbose_, msg)
    }

    virtual std::shared_ptr<gko::LinOp> create()
    {
        auto builder = [this](auto b) {
            return gko::share(b.with_skip_sorting(skip_sorting_)
                                  .with_max_block_size(
                                      static_cast<gko::uint32>(max_block_size_))
                                  .on(exec_));
        };

        auto wrapper = [this](auto f) {
            if (multi_level_schwarz_) {
                auto distmtx =
                    gko::as<RepartDistMatrix>(mtx_)->get_dist_matrix();
                auto local = gko::as<RepartDistMatrix>(mtx_)->get_local();
                auto local_rows = local->get_size()[0];
                return wrap_multi_level_schwarz(distmtx, exec_, f, d_,
                                                local_rows, verbose_);
            } else {
                return wrap_schwarz(mtx_, exec_, std::move(f));
            }
        };

        if (precision_ == "double") {
            return wrapper(builder(dbj::build()));
        }
        if (precision_ == "float") {
            return wrapper(builder(fbj::build()));
        }

        return {};
    }
};
