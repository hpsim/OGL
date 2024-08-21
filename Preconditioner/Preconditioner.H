// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>
#include "DevicePersistent/Base/Base.H"
#include "fvCFD.H"
#include "regIOobject.H"

namespace Foam {
class Preconditioner {
    using mtx = gko::matrix::Csr<scalar>;
    using bj = gko::preconditioner::Jacobi<>;
    using ic = gko::preconditioner::Ic<>;
    using ir = gko::solver::Ir<scalar>;
    using it = gko::stop::Iteration;
    using cg = gko::solver::Cg<scalar>;
    using mg = gko::solver::Multigrid;
    using amgx_pgm = gko::multigrid::Pgm<scalar, label>;
    using ras =
        gko::experimental::distributed::preconditioner::Schwarz<scalar, label,
                                                                label>;

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
              solverControls.lookupOrDefault("preconditionerCaching", 1)),
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
            return gko::share(
                ras::build()
                    .with_generated_local_solver(gko::share(precond->generate(
                        gko::as<gko::experimental::distributed::Matrix<
                            scalar, label, label>>(gkomatrix)
                            ->get_local_matrix())))
                    .on(device_exec)
                    ->generate(gkomatrix));
        }
        return gko::share(precond->generate(gkomatrix));
    }

    template <typename PrecondFactory, typename Factorization>
    std::shared_ptr<gko::LinOp> wrap_schwarz(
        std::shared_ptr<gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec,
        std::unique_ptr<PrecondFactory> precond,
        std::shared_ptr<Factorization> factorization) const
    {
        if (Pstream::parRun()) {
            return gko::share(ras::build()
                                  .with_generated_local_solver(
                                      precond->generate(factorization))
                                  .on(device_exec)
                                  ->generate(gkomatrix));
        }
        return gko::share(precond->generate(factorization));
    }

    std::shared_ptr<gko::LinOp> init_preconditioner_impl(
        const word name, const dictionary &controls,
        std::shared_ptr<gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec) const
    {
        bool skip_sorting =
            controls.lookupOrDefault<Switch>("skipSorting", true);

        if (name == "BJ") {
            // TODO for non constant system matrix reuse block pointers
            label max_block_size(
                controls.lookupOrDefault("maxBlockSize", label(1)));

            word msg = "Generate preconditioner " + name + " MaxBlockSize " +
                       std::to_string(max_block_size);
            MLOG_0(verbose_, msg)

            auto pre_factory = bj::build()
                                   .with_skip_sorting(skip_sorting)
                                   .with_max_block_size(
                                       static_cast<gko::uint32>(max_block_size))
                                   .on(device_exec);
            return wrap_schwarz(gkomatrix, device_exec, std::move(pre_factory));
        }
        if (name == "ILU") {
            word msg = "Generate preconditioner " + name;
            MLOG_0(verbose_, msg)

            auto factorization_factory =
                gko::factorization::Ilu<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .on(device_exec);

            auto factorization = gko::share(factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<
                    scalar, label, label>>(gkomatrix)
                    ->get_local_matrix()));

            auto precond_factory =
                gko::preconditioner::Ilu<>::build().on(device_exec);


            return wrap_schwarz(gkomatrix, device_exec,
                                std::move(precond_factory), factorization);
        }
        if (name == "ILUT") {
            word msg = "Generate preconditioner " + name;
            MLOG_0(verbose_, msg)

            auto factorization_factory =
                gko::factorization::ParIlut<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .on(device_exec);

            auto factorization = gko::share(factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<>>(gkomatrix)
                    ->get_local_matrix()));

            auto precond_factory =
                gko::preconditioner::Ilu<>::build().on(device_exec);


            return wrap_schwarz(gkomatrix, device_exec,
                                std::move(precond_factory), factorization);
        }
        if (name == "IRILU") {
            auto trisolve_factory =
                ir::build()
                    .with_solver(
                        bj::build().with_max_block_size(1u).on(device_exec))
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(5u).on(
                            device_exec))
                    .on(device_exec);

            // Generate an ILU preconditioner factory by setting lower and
            // upper triangular solver - in this case the previously defined
            // iterative refinement method.
            auto precond_factory =
                gko::preconditioner::Ilu<ir, ir>::build()
                    .with_l_solver(gko::clone(trisolve_factory))
                    .with_u_solver(gko::clone(trisolve_factory))
                    .on(device_exec);

            auto factorization_factory =
                gko::factorization::Ilu<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .on(device_exec);

            auto factorization = gko::share(factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<>>(gkomatrix)
                    ->get_local_matrix()));

            // Use incomplete factors to generate ILU preconditioner
            return wrap_schwarz(gkomatrix, device_exec,
                                std::move(precond_factory), factorization);
        }
        if (name == "IC") {
            word msg = "Generate preconditioner " + name;
            MLOG_0(verbose_, msg)

            auto factorization_factory =
                gko::factorization::Ic<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .on(device_exec);

            auto factorization = gko::share(factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<
                    scalar, label, label>>(gkomatrix)
                    ->get_local_matrix()));

            auto precond_factory =
                gko::preconditioner::Ic<>::build().on(device_exec);


            return wrap_schwarz(gkomatrix, device_exec,
                                std::move(precond_factory), factorization);
        }
        if (name == "ICT") {
            bool approx_select(
                controls.lookupOrDefault("approximateSelect", false));
            word msg = "Generate preconditioner " + name +
                       " with approximate select " +
                       std::to_string(approx_select);
            MLOG_0(verbose_, msg)

            auto factorization_factory =
                gko::factorization::ParIct<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .on(device_exec);

            auto ic_factorization = gko::share(factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<
                    scalar, label, label>>(gkomatrix)
                    ->get_local_matrix()));

            auto precond_factory =
                gko::preconditioner::Ic<>::build().on(device_exec);


            return wrap_schwarz(gkomatrix, device_exec,
                                std::move(precond_factory), ic_factorization);
        }
        if (name == "ISAI") {
            label sparsity_power(
                controls.lookupOrDefault("sparsityPower", label(1)));

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
        if (name == "GISAI") {
            label sparsity_power(
                controls.lookupOrDefault("sparsityPower", label(1)));

            word msg = "Generate preconditioner " + name + " SparsityPower " +
                       std::to_string(sparsity_power);
            MLOG_0(verbose_, msg)

            auto pre_factory = gko::preconditioner::Isai<
                                   gko::preconditioner::isai_type::general,
                                   scalar, label>::build()
                                   .with_skip_sorting(skip_sorting)
                                   .with_sparsity_power(sparsity_power)
                                   .on(device_exec);

            return wrap_schwarz(gkomatrix, device_exec, std::move(pre_factory));
        }
        if (name == "Multigrid") {
            word type = controls.lookupOrDefault("type", word("Schwarz"));

            const label coarse_solver_iters(
                controls.lookupOrDefault("coarseSolverIters", label(4)));

            auto cycle_name = controls.lookupOrDefault("cycle", word("v"));
            gko::solver::multigrid::cycle cycle;
            if (cycle_name == "v") cycle = gko::solver::multigrid::cycle::v;
            if (cycle_name == "w") cycle = gko::solver::multigrid::cycle::w;
            if (cycle_name == "f") cycle = gko::solver::multigrid::cycle::f;

            const label max_levels(
                controls.lookupOrDefault("maxLevels", label(5)));
            const label min_coarse_rows(
                controls.lookupOrDefault("minCoarseRows", label(10)));
            bool zeroGuess(controls.lookupOrDefault<Switch>("zeroGuess", true));
            const label smoother_max_iters(
                controls.lookupOrDefault("smootherMaxIters", label(1)));

            auto pre_factory =
                gko::share(bj::build()
                               .with_skip_sorting(true)
                               .with_max_block_size(static_cast<gko::uint32>(1))
                               .on(device_exec));

            auto coarsest_gen = gko::share(
                cg::build()
                    .with_preconditioner(ras::build()
                                             .with_local_solver(pre_factory)
                                             .on(device_exec))
                    .with_criteria(gko::stop::Iteration::build().with_max_iters(
                        static_cast<gko::uint32>(coarse_solver_iters)))
                    .on(device_exec));

            word msg = "Generate preconditioner " + name + " MaxLevels " +
                       std::to_string(max_levels) + " MinCoarseRows " +
                       std::to_string(min_coarse_rows) + " ZeroGuess " +
                       std::to_string(zeroGuess) + " Cycle " + cycle_name +
                       " type: " + type;
            MLOG_0(verbose_, msg)

            if (type == "Schwarz") {
                auto pre_factory =
                    mg::build()
                        .with_max_levels(static_cast<gko::uint32>(max_levels))
                        .with_cycle(cycle)
                        .with_min_coarse_rows(
                            static_cast<gko::uint32>(min_coarse_rows))
                        .with_pre_smoother(
                            ir::build()
                                .with_solver(
                                    bj::build().with_max_block_size(1u))
                                .with_relaxation_factor(0.9)
                                .with_criteria(it::build().with_max_iters(
                                    smoother_max_iters)))
                        .with_post_uses_pre(true)
                        .with_mg_level(
                            amgx_pgm::build().with_deterministic(true).on(
                                device_exec))
                        .with_coarsest_solver(gko::share(
                            cg::build()
                                .with_preconditioner(gko::share(
                                    bj::build()
                                        .with_skip_sorting(true)
                                        .with_max_block_size(
                                            static_cast<gko::uint32>(1))
                                        .on(device_exec)))
                                .with_criteria(
                                    gko::stop::Iteration::build()
                                        .with_max_iters(
                                            static_cast<gko::uint32>(
                                                coarse_solver_iters)))
                                .on(device_exec)))
                        .with_criteria(it::build().with_max_iters(1u))
                        .on(device_exec);

                return wrap_schwarz(gkomatrix, device_exec,
                                    std::move(pre_factory));
            }

            if (type == "Distributed") {
                auto ret = gko::share(
                    gko::solver::Multigrid::build()
                        .with_max_levels(max_levels)
                        .with_mg_level(gko::multigrid::Pgm<scalar>::build()
                                           .with_deterministic(false))
                        .with_min_coarse_rows(min_coarse_rows)
                        .with_coarsest_solver(coarsest_gen)
                        .with_criteria(it::build().with_max_iters(1u))
                        .with_smoother_iters(smoother_max_iters)
                        .with_cycle(cycle)
                        .on(device_exec)
                        ->generate(gkomatrix));
                return ret;
            }
            FatalErrorInFunction << "Unknown Multigrid type: " << type
                                 << "\nValid Choices: Schwarz, Distributed"
                                 << abort(FatalError);


            // std::shared_ptr<gko::LinOp> coarsest_gen;
            // coarsest_gen =
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
        }
        if (name == "none") {
            return {};
        }

        FatalErrorInFunction
            << "OGL does not support the preconditioner: " << name
            << "\nValid Choices: none, BJ, ILU, ISAI, IC, Multigrid"
            << abort(FatalError);
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

#ifdef WITH_ESI_VERSION
        const entry &e =
            solverControls_.lookupEntry("preconditioner", keyType::LITERAL);

        if (e.isDict()) {
            e.dict().readEntry("preconditioner", name);
        } else {
            e.stream() >> name;
        }
#else
        const entry &e =
            solverControls_.lookupEntry("preconditioner", true, true);
        if (e.isDict()) {
            name = e.dict().lookup<word>("preconditioner");
        } else {
            e.stream() >> name;
        }
#endif

        const dictionary &controls = e.isDict() ? e.dict() : dictionary::null;

        auto cache = get_next_caching(sys_matrix_name_, db_);


        if (stored) {
            if (cache > 0) {
                word msg = "Read preconditioner from registry for " +
                           std::to_string(cache);

                LOG_1(verbose_, msg)

                cache = cache - 1;

                set_next_caching(sys_matrix_name_, db_, cache--);
                auto ret =
                    db_.template lookupObjectRef<
                           DevicePersistentBase<gko::LinOp>>(precond_store_name)
                        .get_ptr();
                if (name == "Multigrid") {
                    std::cout << __FILE__ << __LINE__ << "update multigrid\n";
                    word type =
                        controls.lookupOrDefault("type", word("Schwarz"));
                    if (type == "Schwarz") {
                        auto local_solver = std::const_pointer_cast<gko::LinOp>(
                            gko::as<ras>(ret)->get_local_solver());

                        gko::as<gko::UpdateMatrixValue>(local_solver)
                            ->update_matrix_value(
                                gko::as<gko::experimental::distributed::Matrix<
                                    scalar, label, label>>(gkomatrix)
                                    ->get_local_matrix());
                    } else {
                        gko::as<gko::UpdateMatrixValue>(ret)
                            ->update_matrix_value(gkomatrix);
                    }
                }
                return ret;
            } else {
                auto prev_precond = db_.template lookupObjectRef<
                    DevicePersistentBase<gko::LinOp>>(precond_store_name);
                const label caching_period =
                    controls.lookupOrDefault<label>("caching", 0);
                set_next_caching(sys_matrix_name_, db_, caching_period);

                auto generated_precond = init_preconditioner_impl(
                    name, controls, gkomatrix, device_exec);

                auto precond_ptr = prev_precond.get_ptr();
                precond_ptr = generated_precond;
                return precond_ptr;
            }
        }
        const label caching_period =
            controls.lookupOrDefault<label>("caching", 0);
        set_next_caching(sys_matrix_name_, db_, caching_period);
        cache = get_next_caching(sys_matrix_name_, db_);

        auto generated_precond =
            init_preconditioner_impl(name, controls, gkomatrix, device_exec);

        auto po = new DevicePersistentBase<gko::LinOp>(IOobject(path, db_),
                                                       generated_precond);

        // use get_ptr(() to avoid unused variable warning
        po->get_ptr();

        return generated_precond;
    }
};
}  // namespace Foam
