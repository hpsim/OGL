// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <ginkgo/ginkgo.hpp>

namespace Foam {
    template <typename PrecondFactory>
    std::shared_ptr<gko::LinOp> wrap_schwarz(
        std::shared_ptr<const gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec,
        std::unique_ptr<PrecondFactory> precond) const
    {
        auto local = gko::as<RepartDistMatrix>(gkomatrix)->get_local();
        return gko::share(
            ras::build()
                .with_generated_local_solver(precond->generate(local))
                .on(device_exec)
                ->generate(
                    gko::as<RepartDistMatrix>(gkomatrix)->get_dist_mtx()));
    }

    template <typename PrecondFactory, typename Factorization>
    std::shared_ptr<gko::LinOp> wrap_schwarz(
        std::shared_ptr<const gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec,
        std::unique_ptr<PrecondFactory> precond,
        std::shared_ptr<Factorization> factorization) const
    {
        return gko::share(
            ras::build()
                .with_generated_local_solver(precond->generate(factorization))
                .on(device_exec)
                ->generate(gkomatrix));
    }

    }
