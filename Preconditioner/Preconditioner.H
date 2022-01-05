/*---------------------------------------------------------------------------*\
License
    This file is part of OGL.

    OGL is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OGL is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OGL.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::IOSortingIdxHandler

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    IOSortingIdxHandler.H

\*---------------------------------------------------------------------------*/
#ifndef OGL_IOPreconditioner_INCLUDED_H
#define OGL_IOPreconditioner_INCLUDED_H

#include <ginkgo/ginkgo.hpp>
#include "fvCFD.H"
#include "regIOobject.H"

namespace Foam {
class Preconditioner {
    using mtx = gko::matrix::Csr<scalar>;
    using bj = gko::preconditioner::Jacobi<>;
    using ic = gko::preconditioner::Ic<>;

private:
    const word name_;

public:
    Preconditioner(const dictionary &solverControls)
        : name_(
              solverControls.lookupOrDefault("preconditioner", word("none"))){};

    std::shared_ptr<gko::LinOp> init_preconditioner(
        std::shared_ptr<mtx> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec) const
    {
        if (name_ == "BJ") {
            // TODO for non constant system matrix reuse block pointers
            auto pre_factory =
                bj::build().with_skip_sorting(true).with_max_block_size(4u).on(
                    device_exec);
            return gko::share(pre_factory->generate(gkomatrix));
        }
        if (name_ == "ILU") {
            auto factorizaton_factory =
                gko::factorization::Ilu<scalar, label>::build()
                    .with_skip_sorting(true)
                    .on(device_exec);

            auto par_ilu =
                gko::share(factorizaton_factory->generate(gkomatrix));

            auto ilu_pre_factory =
                gko::preconditioner::Ilu<gko::solver::LowerTrs<scalar, label>,
                                         gko::solver::UpperTrs<scalar, label>,
                                         false>::build()
                    .on(device_exec);

            // Use incomplete factors to generate ILU preconditioner auto
            return gko::share(ilu_pre_factory->generate(gko::share(par_ilu)));
        }
        if (name_ == "IC") {
            auto factorizaton_factory =
                gko::factorization::Ic<scalar, label>::build()
                    .with_skip_sorting(true)
                    .on(device_exec);

            auto ic = gko::share(factorizaton_factory->generate(gkomatrix));
            auto pre_factory =
                gko::preconditioner::Ic<>::build().on(device_exec);

            return gko::share(pre_factory->generate(ic));
        }
        if (name_ == "ISAI") {
            auto pre_factory =
                gko::preconditioner::Isai<gko::preconditioner::isai_type::spd,
                                          scalar, label>::build()
                    .with_skip_sorting(true)
                    .on(device_exec);

            return gko::share(pre_factory->generate(gkomatrix));
        }
        if (name_ == "Multigrid") {
            using ir = gko::solver::Ir<scalar>;
            using mg = gko::solver::Multigrid;
            using bj = gko::preconditioner::Jacobi<scalar, label>;
            using amgx_pgm = gko::multigrid::AmgxPgm<scalar, label>;

            auto inner_solver_gen =
                gko::share(bj::build().with_max_block_size(1u).on(device_exec));

            auto smoother_gen = gko::share(
                ir::build()
                    .with_solver(inner_solver_gen)
                    .with_relaxation_factor(0.9)
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(2u).on(
                            device_exec))
                    .on(device_exec));

            // Create MultigridLevel factory
            auto mg_level_gen =
                amgx_pgm::build().with_deterministic(true).on(device_exec);

            // Create CoarsestSolver factory
            auto coarsest_gen = gko::share(
                ir::build()
                    .with_solver(inner_solver_gen)
                    .with_relaxation_factor(0.9)
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(4u).on(
                            device_exec))
                    .on(device_exec));

            // Create multigrid factory
            auto pre_factory =
                mg::build()
                    .with_max_levels(9u)
                    .with_min_coarse_rows(10u)
                    .with_pre_smoother(smoother_gen)
                    .with_post_uses_pre(true)
                    .with_mg_level(gko::share(mg_level_gen))
                    .with_coarsest_solver(coarsest_gen)
                    .with_zero_guess(true)
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u).on(
                            device_exec))
                    .on(device_exec);

            return gko::share(pre_factory->generate(gkomatrix));
        }

        return {};
    };
};
}  // namespace Foam

#endif