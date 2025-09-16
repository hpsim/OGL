// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/Preconditioner/CoarseSolver.hpp"
#include "OGL/common.hpp"

std::shared_ptr<const gko::LinOpFactory> generate_coarse_solver(
    std::shared_ptr<gko::Executor> exec, const dictionary &d, label verbose)
{
    using dbj = gko::preconditioner::Jacobi<double, label>;
    using cg = gko::solver::Cg<scalar>;
    using bicgstab = gko::solver::Bicgstab<scalar>;
    using gmres = gko::solver::Gmres<scalar>;
    using ir = gko::solver::Ir<scalar>;
    using ras =
        gko::experimental::distributed::preconditioner::Schwarz<scalar, label,
                                                                label>;

    auto hasSubDict = d.found("coarseSolverConfig");
    auto solverDict =
        (hasSubDict) ? d.subDict("coarseSolverConfig") : dictionary{};

    word solver = solverDict.lookupOrDefault("solver", word("CG"));
    scalar solveNorm = solverDict.lookupOrDefault("relTol", scalar(1e-6));
    label maxIter = solverDict.lookupOrDefault("maxIter", label(1));

    word msg = "\nGenerate coarse solver\n\tsolver: " + solver +
               "\n\trelTolCoarse: " + std::to_string(solveNorm) +
               "\n\tmaxIter: " + std::to_string(maxIter);
    MLOG_0(verbose, msg)

    auto solve_it = gko::stop::Iteration::build().with_max_iters(
        static_cast<gko::uint32>(maxIter));
    auto solve_red =
        gko::stop::ResidualNorm<scalar>::build().with_reduction_factor(
            solveNorm);

    std::shared_ptr<const gko::LinOpFactory> coarsest_solver = {};
    auto bjfac = ras::build().with_local_solver(
        dbj::build().with_skip_sorting(true).with_max_block_size(1u).on(exec));

    if (solver == "CG") {
        coarsest_solver = gko::share(cg::build()
                                         .with_preconditioner(bjfac)
                                         .with_criteria(solve_it, solve_red)
                                         .on(exec));
    }
    if (solver == "BiCGStab") {
        coarsest_solver = gko::share(bicgstab::build()
                                         .with_preconditioner(bjfac)
                                         .with_criteria(solve_it, solve_red)
                                         .on(exec));
    }
    if (solver == "GMRES") {
        coarsest_solver = gko::share(gmres::build()
                                         .with_preconditioner(bjfac)
                                         .with_criteria(solve_it, solve_red)
                                         .on(exec));
    }
    if (solver == "Jacobi") {
        scalar relaxFac{
            solverDict.lookupOrDefault("relaxationFactor", scalar(0.9))};
        coarsest_solver = gko::share(ir::build()
                                         .with_solver(bjfac)
                                         .with_relaxation_factor(relaxFac)
                                         .with_criteria(solve_it, solve_red)
                                         .on(exec));
    }
    if (coarsest_solver == nullptr) {
        FatalErrorInFunction << "Unknown coarse solver: " << solver
                             << "\nValid choices: CG, BiCGStab, GMRES, Jacobi"
                             << abort(FatalError);
    }
    return coarsest_solver;
}
