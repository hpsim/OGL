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
    std::shared_ptr<PrecondFactory> precond)
{
    using ras =
        gko::experimental::distributed::preconditioner::Schwarz<scalar, label,
                                                                label>;

    std::cout << __FILE__ << __LINE__ << " MG 3\n";
    auto local = gko::as<RepartDistMatrix>(gkomatrix)->get_local();
    std::cout << __FILE__ << __LINE__ << " MG 4\n";
    return gko::share(
        ras::build()
            .with_generated_local_solver(precond->generate(local))
            .on(device_exec)
            ->generate(gko::as<RepartDistMatrix>(gkomatrix)->get_dist_mtx()));
}

template <typename PrecondFactory, typename Factorization>
std::shared_ptr<gko::LinOp> wrap_schwarz(
    std::shared_ptr<const gko::LinOp> gkomatrix,
    std::shared_ptr<gko::Executor> device_exec,
    std::shared_ptr<PrecondFactory> precond,
    std::shared_ptr<Factorization> factorization)
{
    using ras =
        gko::experimental::distributed::preconditioner::Schwarz<scalar, label,
                                                                label>;
    return gko::share(
        ras::build()
            .with_generated_local_solver(precond->generate(factorization))
            .on(device_exec)
            ->generate(gkomatrix));
}


template <typename PrecondFactory>
std::shared_ptr<gko::LinOp> wrap_multi_level_schwarz(
    std::shared_ptr<const gko::LinOp> gkomatrix,
    std::shared_ptr<gko::Executor> device_exec,
    std::shared_ptr<PrecondFactory> precond, const dictionary &d,
    label local_rows, label verbose)
{
    using pgm = gko::multigrid::Pgm<scalar, label>;
    using fc = gko::multigrid::FixedCoarsening<scalar, label>;
    using bj = gko::preconditioner::Jacobi<scalar, label>;
    using solver = gko::solver::Cg<scalar>;
    using ras =
        gko::experimental::distributed::preconditioner::Schwarz<scalar, label,
                                                                label>;

    auto selCoarseRows = d.lookupOrDefault("selCoarseRows", label(5));
    auto fixedCoarsening = d.lookupOrDefault<Switch>("fixedCoarsening", false);
    auto coarseWeight = d.lookupOrDefault("coarseWeight", scalar(0.01));
    auto solveNormC = d.lookupOrDefault("reductionCoarseSolver", label(1e-6));
    auto maxIterCoarse = d.lookupOrDefault("maxIterCoarse", label(50));

    word msg = "Generate multi level schwarz:\n\tfixedCoarsening " +
               std::to_string(fixedCoarsening) + "\n\tselCoarseRows " +
               std::to_string(selCoarseRows) + "\n\trelTolCoarse " +
               std::to_string(solveNormC) + "\n\tmaxIterCoarse " +
               std::to_string(maxIterCoarse) + "\n\tcoarseWeigth" +
               std::to_string(coarseWeight);
    MLOG_0(verbose, msg)

    auto pre_factory = ras::build().with_local_solver(
        bj::build().with_skip_sorting(true).with_max_block_size(1u).on(
            device_exec));

    auto coarse_solver = gko::share(
        solver::build()
            .with_preconditioner(pre_factory)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(maxIterCoarse),
                gko::stop::ResidualNorm<scalar>::build().with_reduction_factor(
                    solveNormC))
            .on(device_exec));

    if (fixedCoarsening) {
        auto n_rows = local_rows / selCoarseRows;
        auto sel_rows =
            gko::array<label>(gko::ReferenceExecutor::create(), n_rows);
        for (auto i = 0; i < sel_rows.get_size(); i++) {
            sel_rows.get_data()[i] = selCoarseRows * i;
        }

        sel_rows.set_executor(device_exec);
        auto pgm_fac = gko::share(
            fc::build().with_skip_sorting(true).with_coarse_rows(sel_rows).on(
                device_exec));

        return gko::share(ras::build()
                              .with_local_solver(precond)
                              .with_coarse_level(pgm_fac)
                              .with_l1_smoother(false)
                              .with_coarse_solver(coarse_solver)
                              .with_coarse_weight(coarseWeight)
                              .on(device_exec)
                              ->generate(gkomatrix));
    } else {
        auto pgm_fac =
            gko::share(pgm::build().with_skip_sorting(true).on(device_exec));

        return gko::share(ras::build()
                              .with_local_solver(precond)
                              .with_coarse_level(pgm_fac)
                              .with_l1_smoother(false)
                              .with_coarse_solver(coarse_solver)
                              .on(device_exec)
                              ->generate(gkomatrix));
    }
}

}  // namespace Foam
