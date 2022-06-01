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
#include "../DevicePersistentBase/DevicePersistentBase.H"
#include "fvCFD.H"
#include "regIOobject.H"

namespace Foam {
class Preconditioner {
    using ras = gko::preconditioner::Schwarz<scalar, label>;
    using mtx = gko::matrix::Csr<scalar>;
    using bj = gko::preconditioner::Jacobi<>;
    using ic = gko::preconditioner::Ic<>;
    using ir = gko::solver::Ir<scalar>;

private:
    const word sys_matrix_name_;

    const objectRegistry &db_;

    const label cache_preconditioner_;

    const dictionary &solverControls_;

    const label verbose_;

public:
    Preconditioner(const word sys_matrix_name, const objectRegistry &db,
                   const dictionary &solverControls, const label verbose)
        : sys_matrix_name_(sys_matrix_name),
          db_(db),
          cache_preconditioner_(
              solverControls.lookupOrDefault("PreconditionerCaching", 1)),
          solverControls_(solverControls),
          verbose_(verbose)
    {}

    template <typename PrecondFactory>
    std::shared_ptr<gko::LinOp> wrap_schwarz(
        std::shared_ptr<gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec,
        std::unique_ptr<PrecondFactory> precond) const
    {
        if (Pstream::parRun()) {
            return gko::share(ras::build()
                                  .with_inner_solver(std::move(precond))
                                  .on(device_exec)
                                  ->generate(gkomatrix));
        }
        return gko::share(precond->generate(gkomatrix));
    }

    std::shared_ptr<gko::LinOp> update_preconditioner(
        const word name, const dictionary &controls,
        std::shared_ptr<gko::LinOp> prev_precond,
        std::shared_ptr<gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec) const
    {
        bool skip_sorting =
            controls.lookupOrDefault<Switch>("SkipSorting", true);

        if (name == "BJ") {
            // TODO for non constant system matrix reuse block pointers
            label max_block_size(
                controls.lookupOrDefault("MaxBlockSize", label(1)));

            word msg = "Update preconditioner " + name + " MaxBlockSize " +
                       std::to_string(max_block_size);
            MLOG_0(verbose_, msg)

            std::cout << "copy prev block ptr" << std::endl;
            auto prev_block_ptr = gko::array<label>(
                device_exec,
                // gko::as<bj>(gko::as<ras>(prev_precond)->get_inner_solvers()[0])
                //     ->get_parameters()
                //     .block_pointers.get_num_elems(),
                gko::as<bj>(gko::as<ras>(prev_precond)->get_inner_solvers()[0])
                    ->get_parameters()
                    .block_pointers);
            std::cout << "copy prev block ptr done" << std::endl;

            auto pre_factory = bj::build()
                                   .with_skip_sorting(skip_sorting)
                                   .with_max_block_size(max_block_size)
                                   .with_block_pointers(prev_block_ptr)
                                   .on(device_exec);


            std::cout << "prefactory" << std::endl;
            auto sw =
                wrap_schwarz(gkomatrix, device_exec, std::move(pre_factory));
            std::cout << "prefactory done" << std::endl;
            return sw;
        }
        return prev_precond;
    }

    std::shared_ptr<gko::LinOp> init_preconditioner_impl(
        const word name, const dictionary &controls,
        std::shared_ptr<gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec) const
    {
        bool skip_sorting =
            controls.lookupOrDefault<Switch>("SkipSorting", true);

        if (name == "BJ") {
            // TODO for non constant system matrix reuse block pointers
            label max_block_size(
                controls.lookupOrDefault("MaxBlockSize", label(1)));

            word msg = "Generate preconditioner " + name + " MaxBlockSize " +
                       std::to_string(max_block_size);
            MLOG_0(verbose_, msg)

            auto pre_factory = bj::build()
                                   .with_skip_sorting(skip_sorting)
                                   .with_max_block_size(max_block_size)
                                   .on(device_exec);
            return wrap_schwarz(gkomatrix, device_exec, std::move(pre_factory));
        }
        if (name == "ILU") {
            auto factorizaton_factory =
                gko::factorization::Ilu<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .on(device_exec);

            word msg = "Generate preconditioner " + name;
            MLOG_0(verbose_, msg)

            auto ilu_pre_factory =
                gko::preconditioner::Ilu<gko::solver::LowerTrs<scalar, label>,
                                         gko::solver::UpperTrs<scalar, label>,
                                         false>::build()
                    .on(device_exec);

            // Use incomplete factors to generate ILU preconditioner auto
            return wrap_schwarz(
                gko::share(factorizaton_factory->generate(gkomatrix)),
                device_exec, std::move(ilu_pre_factory));
        }
        if (name == "IRILU") {
            auto trisolve_factory =
                ir::build()
                    .with_solver(
                        bj::build().with_max_block_size(1).on(device_exec))
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(5).on(
                            device_exec))
                    .on(device_exec);

            // Generate an ILU preconditioner factory by setting lower and
            // upper triangular solver - in this case the previously defined
            // iterative refinement method.
            auto ilu_pre_factory =
                gko::preconditioner::Ilu<ir, ir>::build()
                    .with_l_solver_factory(gko::clone(trisolve_factory))
                    .with_u_solver_factory(gko::clone(trisolve_factory))
                    .on(device_exec);

            auto factorizaton_factory =
                gko::factorization::Ilu<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .on(device_exec);

            // Use incomplete factors to generate ILU preconditioner
            return wrap_schwarz(
                gko::share(factorizaton_factory->generate(gkomatrix)),
                device_exec, std::move(ilu_pre_factory));
        }
        if (name == "IC") {
            auto factorizaton_factory =
                gko::factorization::Ic<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .on(device_exec);

            word msg = "Generate preconditioner " + name;
            MLOG_0(verbose_, msg)

            auto ic = gko::share(factorizaton_factory->generate(gkomatrix));
            auto pre_factory =
                gko::preconditioner::Ic<>::build().on(device_exec);

            return wrap_schwarz(gkomatrix, device_exec, std::move(pre_factory));
        }
        if (name == "ISAI") {
            label sparsity_power(
                controls.lookupOrDefault("SparsityPower", label(1)));

            word msg = "Generate preconditioner " + name + " SparsityPower " +
                       std::to_string(sparsity_power);
            MLOG_0(verbose_, msg)

            auto pre_factory =
                gko::preconditioner::Isai<gko::preconditioner::isai_type::spd,
                                          scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .with_sparsity_power(sparsity_power)
                    .on(device_exec);

            return wrap_schwarz(gkomatrix, device_exec, std::move(pre_factory));
        }
        if (name == "Multigrid") {
            using ir = gko::solver::Ir<scalar>;
            using cg = gko::solver::Cg<scalar>;
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
            // std::shared_ptr<gko::LinOp> coarsest_gen;
            // coarsest_gen =
            //
            // (solverControls_.lookupOrDefault("PreconditionerMultigridUseIR",
            //                                      word("IR")) == "IR")
            //         ? gko::share(
            //               ir::build()
            //                   .with_solver(inner_solver_gen)
            //                   .with_relaxation_factor(0.9)
            //                   .with_criteria(gko::stop::Iteration::build()
            //                                      .with_max_iters(4u)
            //                                      .on(device_exec))
            //                   .on(device_exec))
            //         : gko::share(
            //               cg::build()
            //                   .with_criteria(gko::stop::Iteration::build()
            //                                      .with_max_iters(4u)
            //                                      .on(device_exec))
            //                   .on(device_exec));

            auto coarsest_gen = gko::share(
                cg::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(4u).on(
                            device_exec))
                    .on(device_exec));

            // Create multigrid factory
            label max_levels(controls.lookupOrDefault("MaxLevels", label(9)));
            label min_coarse_rows(
                controls.lookupOrDefault("MinCoarseRows", label(10)));
            bool zeroGuess(controls.lookupOrDefault<Switch>("ZeroGuess", true));

            word msg = "Generate preconditioner " + name + " MaxLevels " +
                       std::to_string(max_levels) + " MinCoarseRows " +
                       std::to_string(min_coarse_rows) + " ZeroGuess " +
                       std::to_string(zeroGuess);
            MLOG_0(verbose_, msg)

            auto pre_factory =
                mg::build()
                    .with_max_levels(max_levels)
                    .with_min_coarse_rows(min_coarse_rows)
                    .with_pre_smoother(smoother_gen)
                    .with_post_uses_pre(true)
                    .with_mg_level(
                        amgx_pgm::build().with_deterministic(true).on(
                            device_exec))
                    .with_coarsest_solver(coarsest_gen)
                    .with_zero_guess(zeroGuess)
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u).on(
                            device_exec))
                    .on(device_exec);
            return wrap_schwarz(gkomatrix, device_exec, std::move(pre_factory));
        }
        return {};
    }

    std::shared_ptr<gko::LinOp> init_preconditioner(
        std::shared_ptr<gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec) const
    {
        const word precond_store_name = "Cached_preconditinoner";
        const fileName path = precond_store_name;
        bool stored{db_.template foundObject<regIOobject>(precond_store_name)};

        word name;

        const entry &e =
            solverControls_.lookupEntry("preconditioner", keyType::LITERAL);

        if (e.isDict()) {
            e.dict().readEntry("preconditioner", name);
        } else {
            e.stream() >> name;
        }

        const dictionary &controls = e.isDict() ? e.dict() : dictionary::null;

        auto cache = get_next_caching(sys_matrix_name_, db_);


        if (stored) {
            if (cache > 0) {
                word msg = "Read preconditioner from registry for " +
                           std::to_string(cache);

                LOG_1(verbose_, msg)

                cache = cache - 1;

                set_next_caching(sys_matrix_name_, db_, cache--);
                return db_
                    .template lookupObjectRef<DevicePersistentBase<gko::LinOp>>(
                        precond_store_name)
                    .get_ptr();
            } else {
                auto prev_precond = db_.template lookupObjectRef<
                    DevicePersistentBase<gko::LinOp>>(precond_store_name);
                const label caching_period =
                    controls.lookupOrDefault<label>("Caching", 0);
                set_next_caching(sys_matrix_name_, db_, caching_period);
                auto precond_ptr = prev_precond.get_ptr();

                const bool update =
                    controls.lookupOrDefault<Switch>("Update", false);

                if (update) {
                    auto updated_precond = update_preconditioner(
                        name, controls, prev_precond.get_ptr(), gkomatrix,
                        device_exec);
                    precond_ptr = updated_precond;
                    return precond_ptr;
                } else {
                    auto generated_precond = init_preconditioner_impl(
                        name, controls, gkomatrix, device_exec);

                    precond_ptr = generated_precond;
                    return precond_ptr;
                }
            }
        }
        const label caching_period =
            controls.lookupOrDefault<label>("Caching", 0);
        set_next_caching(sys_matrix_name_, db_, caching_period);
        cache = get_next_caching(sys_matrix_name_, db_);

        auto generated_precond =
            init_preconditioner_impl(name, controls, gkomatrix, device_exec);

        auto po = new DevicePersistentBase<gko::LinOp>(IOobject(path, db_),
                                                       generated_precond);

        return generated_precond;
    };
};
}  // namespace Foam

#endif
