// SPDX-FileCopyrightText: 2025 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "PreconditionerWrapper.hpp"
#include "OGL/Preconditioner/Schwarz.hpp"

class BJ : public PreconditionerWrapper {
    using dbj = gko::preconditioner::Jacobi<double, label>;
    using fbj = gko::preconditioner::Jacobi<float, label>;

    const label verbose_;
    bool skip_sorting_;
    label max_block_size_;
    word precision_;


public:
    BJ(const dictionary &d, label verbose)
        : verbose_(verbose),
          skip_sorting_(d.lookupOrDefault<Switch>("skipSorting", true)),
          max_block_size_(d.lookupOrDefault("maxBlockSize", label(1))),
          precision_(d.lookupOrDefault("precision", word("double")))
    {
            word msg = "Generate (Block)-Jacobi preconditioner:"
                       + "\n\tprecision: " precision +
                       "\n\tmaxBlockSize " + std::to_string(max_block_size);
            MLOG_0(verbose_, msg)
    }

    virtual std::shared_ptr<gko::LinOp> create_local()
    {
        // auto pre_factory =
        //     dbj::build()
        //         .with_skip_sorting(skip_sorting)
        //         .with_max_block_size(
        //             static_cast<gko::uint32>(max_block_size))
        //         .on(device_exec);
        // auto gkodistmatrix =
        //     gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
        // return wrap_schwarz(gkomatrix, device_exec,
        //                     std::move(pre_factory));
    }


};
