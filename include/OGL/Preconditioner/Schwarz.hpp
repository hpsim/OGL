// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <ginkgo/ginkgo.hpp>
#include "OGL/Preconditioner/CoarseSolver.hpp"

namespace Foam {
template <typename PrecondFactory>
std::shared_ptr<gko::LinOp> wrap_schwarz(
    std::shared_ptr<const gko::LinOp> gkomatrix,
    std::shared_ptr<gko::Executor> device_exec,
    std::shared_ptr<PrecondFactory> precond)
{
    using ras =
        gko::experimental::distributed::preconditioner::Schwarz<scalar, label,
                                                                label>;
    // auto local = gko::as<RepartDistMatrix>(gkomatrix)->get_local();
    return gko::share(
        ras::build()
            .with_generated_local_solver(precond)
            .on(device_exec)
            ->generate(gko::as<RepartDistMatrix>(gkomatrix)->get_dist_mtx()));
}

// template <typename PrecondFactory, typename Factorization>
// std::shared_ptr<gko::LinOp> wrap_schwarz(
//     std::shared_ptr<const gko::LinOp> gkomatrix,
//     std::shared_ptr<gko::Executor> device_exec,
//     std::shared_ptr<PrecondFactory> precond,
//     std::shared_ptr<Factorization> factorization)
// {
//     using ras =
//         gko::experimental::distributed::preconditioner::Schwarz<scalar,
//         label,
//                                                                 label>;
//     return gko::share(
//         ras::build()
//             .with_generated_local_solver(precond->generate(factorization))
//             .on(device_exec)
//             ->generate(gkomatrix));
// }
//

template <typename Precond>
std::shared_ptr<gko::LinOp> wrap_multi_level_schwarz(
    std::shared_ptr<const gko::LinOp> mtx,
    std::shared_ptr<gko::Executor> device_exec,
    std::shared_ptr<Precond> precond, const dictionary &d, label verbose)
{
    using pgm = gko::multigrid::Pgm<scalar, label>;
    using fc = gko::multigrid::FixedCoarsening<scalar, label>;
    using bj = gko::preconditioner::Jacobi<scalar, label>;
    using solver = gko::solver::Cg<scalar>;
    using ras =
        gko::experimental::distributed::preconditioner::Schwarz<scalar, label,
                                                                label>;

    auto distmtx = gko::as<RepartDistMatrix>(mtx)->get_dist_matrix();
    auto local_rows =
        gko::as<RepartDistMatrix>(mtx)->get_local()->get_size()[0];

    auto coarse_solver = generate_coarse_solver(device_exec, d, verbose);
    auto coarsening = d.lookupOrDefault<word>("coarsening", word("PGM"));
    auto coarseWeight = d.lookupOrDefault("coarseWeight", scalar(0.1));

    if (coarsening == "fixed") {
        auto selCoarseRows = d.lookupOrDefault("selCoarseRows", label(5));
        auto solveNormC =
            d.lookupOrDefault("reductionCoarseSolver", label(1e-6));
        auto maxIterCoarse = d.lookupOrDefault("maxIterCoarse", label(50));
        word msg =
            "\nGenerate multi level schwarz:\n\tcoarsening: "
            "fixed\n\tselCoarseRows: " +
            std::to_string(selCoarseRows) +
            "\n\tcoarseWeight: " + std::to_string(coarseWeight);
        MLOG_0(verbose, msg)

        auto n_rows = local_rows / selCoarseRows;
        auto sel_rows =
            gko::array<label>(gko::ReferenceExecutor::create(), n_rows);
        for (auto i = 0; i < sel_rows.get_size(); i++) {
            sel_rows.get_data()[i] = selCoarseRows * i;
        }
        sel_rows.set_executor(device_exec);
        auto coarsening_fac = gko::share(
            fc::build().with_skip_sorting(true).with_coarse_rows(sel_rows).on(
                device_exec));

        return gko::share(ras::build()
                              .with_generated_local_solver(precond)
                              .with_coarse_level(coarsening_fac)
                              .with_l1_smoother(false)
                              .with_coarse_solver(coarse_solver)
                              .with_coarse_weight(coarseWeight)
                              .on(device_exec)
                              ->generate(distmtx));
    }
    if (coarsening == "PGM") {
        word msg =
            "\nGenerate multi level schwarz:\n\tfixedCoarsening: "
            "PGM\n\tcoarseWeight: " +
            std::to_string(coarseWeight);
        MLOG_0(verbose, msg)
        auto pgm_fac =
            gko::share(pgm::build().with_skip_sorting(true).on(device_exec));
        return gko::share(ras::build()
                              .with_generated_local_solver(precond)
                              .with_coarse_level(pgm_fac)
                              .with_l1_smoother(false)
                              .with_coarse_weight(coarseWeight)
                              .with_coarse_solver(coarse_solver)
                              .on(device_exec)
                              ->generate(distmtx));
    }
}


template <typename PrecondFactory>
std::shared_ptr<gko::LinOp> dispatch_schwarz(
    std::shared_ptr<const gko::LinOp> mtx, std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<PrecondFactory> precond, const dictionary &d, label verbose)
{
    if (d.lookupOrDefault<Switch>("multiLevelSchwarz", false)) {
        return wrap_multi_level_schwarz(mtx, exec, precond,
                                        d.subDict("multiLevelConfig"), verbose);
    } else {
        return wrap_schwarz(mtx, exec, std::move(precond));
    }
}

}  // namespace Foam
