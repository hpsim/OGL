// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "OGL/DevicePersistent/Base.hpp"
#include "OGL/MatrixWrapper/Distributed.hpp"

#include "fvCFD.H"
#include "regIOobject.H"

namespace Foam {
class Preconditioner {
    using mtx = gko::matrix::Csr<scalar>;
    using bj = gko::preconditioner::Jacobi<scalar, label>;
    using fbj = gko::preconditioner::Jacobi<float, label>;
    using dbj = gko::preconditioner::Jacobi<double, label>;
    using sor = gko::preconditioner::Sor<scalar, label>;
    using ic = gko::preconditioner::Ic<>;
    using ir = gko::solver::Ir<scalar>;
    using it = gko::stop::Iteration;
    using cg = gko::solver::Cg<scalar>;
    using fcg = gko::solver::Cg<float>;
    using mg = gko::solver::Multigrid;
    using pgm = gko::multigrid::Pgm<scalar, label>;
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
    std::shared_ptr<gko::LinOp> wrap_multi_level_schwarz(
        std::shared_ptr<const gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec,
        std::shared_ptr<PrecondFactory> precond, const dictionary &d,
        label local_rows) const
    {
        using pgm = gko::multigrid::Pgm<scalar, label>;
        using fc = gko::multigrid::FixedCoarsening<scalar, label>;
        using solver = gko::solver::Cg<scalar>;

        auto selCoarseRows = d.lookupOrDefault("selCoarseRows", label(5));
        auto fixedCoarsening =
            d.lookupOrDefault<Switch>("fixedCoarsening", false);
        auto coarseWeight = d.lookupOrDefault("coarseWeight", scalar(0.01));
        auto solveNormC =
            d.lookupOrDefault("reductionCoarseSolver", label(1e-6));
        auto maxIterCoarse = d.lookupOrDefault("maxIterCoarse", label(50));

        word msg = "Generate multi level schwarz:\n\tfixedCoarsening " +
                   std::to_string(fixedCoarsening) + "\n\tselCoarseRows " +
                   std::to_string(selCoarseRows) + "\n\trelTolCoarse " +
                   std::to_string(solveNormC) + "\n\tmaxIterCoarse " +
                   std::to_string(maxIterCoarse) + "\n\tcoarseWeigth" +
                   std::to_string(coarseWeight);
        MLOG_0(verbose_, msg)

        auto pre_factory = ras::build().with_local_solver(
            bj::build().with_skip_sorting(true).with_max_block_size(1u).on(
                device_exec));

        auto coarse_solver = gko::share(
            solver::build()
                .with_preconditioner(pre_factory)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(maxIterCoarse),
                    gko::stop::ResidualNorm<scalar>::build()
                        .with_reduction_factor(solveNormC))
                .on(device_exec));

        if (fixedCoarsening) {
            auto n_rows = local_rows / selCoarseRows;
            auto sel_rows =
                gko::array<label>(gko::ReferenceExecutor::create(), n_rows);
            for (auto i = 0; i < sel_rows.get_size(); i++) {
                sel_rows.get_data()[i] = selCoarseRows * i;
            }

            sel_rows.set_executor(device_exec);
            auto pgm_fac = gko::share(fc::build()
                                          .with_skip_sorting(true)
                                          .with_coarse_rows(sel_rows)
                                          .on(device_exec));

            return gko::share(ras::build()
                                  .with_local_solver(precond)
                                  .with_coarse_level(pgm_fac)
                                  .with_l1_smoother(false)
                                  .with_coarse_solver(coarse_solver)
                                  .with_coarse_weight(coarseWeight)
                                  .on(device_exec)
                                  ->generate(gkomatrix));
        } else {
            auto pgm_fac = gko::share(
                pgm::build().with_skip_sorting(true).on(device_exec));

            return gko::share(ras::build()
                                  .with_local_solver(precond)
                                  .with_coarse_level(pgm_fac)
                                  .with_l1_smoother(false)
                                  .with_coarse_solver(coarse_solver)
                                  .on(device_exec)
                                  ->generate(gkomatrix));
        }
    }

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

    std::shared_ptr<gko::LinOp> init_preconditioner_impl(
        const word name, const dictionary &d,
        std::shared_ptr<const gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec) const
    {
        bool skip_sorting = d.lookupOrDefault<Switch>("skipSorting", true);
        bool multi_level_schwarz =
            d.lookupOrDefault<Switch>("multiLevelSchwarz", false);

        if (name == "BJ") {
            // TODO for non constant system matrix reuse block pointers
            label max_block_size(d.lookupOrDefault("maxBlockSize", label(1)));
            word precision(d.lookupOrDefault("precision", word("double")));

            word msg = "Generate preconditioner " + name + "<" + precision +
                       "> MaxBlockSize " + std::to_string(max_block_size);
            MLOG_0(verbose_, msg)

            if (precision == "double") {
                if (multi_level_schwarz) {
                    auto pre_factory = gko::share(
                        dbj::build()
                            .with_skip_sorting(skip_sorting)
                            .with_max_block_size(
                                static_cast<gko::uint32>(max_block_size))
                            .on(device_exec));
                    auto gkodistmatrix =
                        gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
                    auto local =
                        gko::as<RepartDistMatrix>(gkomatrix)->get_local();
                    auto local_rows = local->get_size()[0];
                    return wrap_multi_level_schwarz(gkodistmatrix, device_exec,
                                                    pre_factory, d, local_rows);
                } else {
                    auto pre_factory =
                        dbj::build()
                            .with_skip_sorting(skip_sorting)
                            .with_max_block_size(
                                static_cast<gko::uint32>(max_block_size))
                            .on(device_exec);
                    auto gkodistmatrix =
                        gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
                    return wrap_schwarz(gkomatrix, device_exec,
                                        std::move(pre_factory));
                }
            } else {
                auto pre_factory =
                    fbj::build()
                        .with_skip_sorting(skip_sorting)
                        .with_max_block_size(
                            static_cast<gko::uint32>(max_block_size))
                        .on(device_exec);
                return wrap_schwarz(gkomatrix, device_exec,
                                    std::move(pre_factory));
            }
        }
        if (name == "ILU") {
            label iterations(d.lookupOrDefault("iterations", label(0)));
            word msg = "Generate preconditioner " + name;
            MLOG_0(verbose_, msg)

            auto factorization_factory =
                gko::factorization::ParIlu<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .with_iterations(iterations)
                    .on(device_exec);
            auto gkodistmatrix =
                gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
            auto factorization = gko::share(factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<
                    scalar, label, label>>(gkodistmatrix)
                    ->get_local_matrix()));
            auto precond_factory =
                gko::preconditioner::Ilu<>::build().on(device_exec);
            return wrap_schwarz(gkodistmatrix, device_exec,
                                std::move(precond_factory), factorization);
        }
        if (name == "ILUT") {
            word msg = "Generate preconditioner " + name;
            MLOG_0(verbose_, msg)
            auto gkodistmatrix =
                gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();

            auto factorization_factory =
                gko::factorization::ParIlut<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .on(device_exec);

            auto factorization = gko::share(factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<>>(gkodistmatrix)
                    ->get_local_matrix()));
            auto precond_factory =
                gko::preconditioner::Ilu<>::build().on(device_exec);
            return wrap_schwarz(gkodistmatrix, device_exec,
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
            auto gkodistmatrix =
                gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
            auto factorization = gko::share(factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<
                    scalar, label, label>>(gkodistmatrix)
                    ->get_local_matrix()));
            auto precond_factory =
                gko::preconditioner::Ic<>::build().on(device_exec);
            return wrap_schwarz(gkodistmatrix, device_exec,
                                std::move(precond_factory), factorization);
        }

        if (name == "ICT") {
            bool approx_select(d.lookupOrDefault("approximateSelect", true));
            word msg = "Generate preconditioner " + name +
                       " with approximate select " +
                       std::to_string(approx_select);
            MLOG_0(verbose_, msg)

            auto factorization_factory =
                gko::factorization::ParIct<scalar, label>::build()
                    .with_skip_sorting(skip_sorting)
                    .on(device_exec);

            auto gkodistmatrix =
                gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
            auto ic_factorization = gko::share(factorization_factory->generate(
                gko::as<gko::experimental::distributed::Matrix<
                    scalar, label, label>>(gkodistmatrix)
                    ->get_local_matrix()));

            auto precond_factory =
                gko::preconditioner::Ic<>::build().on(device_exec);

            return wrap_schwarz(gkomatrix, device_exec,
                                std::move(precond_factory), ic_factorization);
        }
        if (name == "ISAI") {
            label sparsity_power(d.lookupOrDefault("sparsityPower", label(1)));

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
            label sparsity_power(d.lookupOrDefault("sparsityPower", label(1)));

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
            word type = d.lookupOrDefault("type", word("Schwarz"));

            auto maxIterCoarseS(d.lookupOrDefault("maxIterCoarse", label(4)));
            auto solveNorm = d.lookupOrDefault("relTolCoarse", scalar(1e-6));
            auto relaxFac = d.lookupOrDefault("relaxationFactor", scalar(0.9));
            auto cycleName = d.lookupOrDefault("cycle", word("v"));
            auto maxLevels = d.lookupOrDefault("maxLevels", label(5));
            auto minRowsC = d.lookupOrDefault("minCoarseRows", label(10));
            auto smoother = d.lookupOrDefault("smoother", word("Jacobi"));
            auto maxIterS = d.lookupOrDefault("maxIterSmoother", label(1));

            gko::solver::multigrid::cycle cycle;
            if (cycleName == "v") cycle = gko::solver::multigrid::cycle::v;
            if (cycleName == "w") cycle = gko::solver::multigrid::cycle::w;
            if (cycleName == "f") cycle = gko::solver::multigrid::cycle::f;

            word msg = "Generate preconditioner: " + name +
                       "\n\tmaxLevels: " + std::to_string(maxLevels) +
                       "\n\tminCoarseRows: " + std::to_string(minRowsC) +
                       "\n\tSmoother: " + smoother +
                       "\n\trelaxationFactor: " + std::to_string(relaxFac) +
                       "\n\tmaxIterSmoother: " + std::to_string(maxIterS) +
                       "\n\tmaxIterCoarse: " + std::to_string(maxIterCoarseS) +
                       "\n\tinnerSolverNorm: " + std::to_string(solveNorm) +
                       "\n\tcycle: " + cycleName + " type: " + type;
            MLOG_0(verbose_, msg)


            std::shared_ptr<gko::LinOpFactory> bjfac{};

            if (smoother == "Jacobi") {
                bjfac = bj::build()
                            .with_max_block_size(1u)
                            .with_skip_sorting(true)
                            .on(device_exec);
            }
            if (smoother == "SOR") {
                bjfac = sor::build()
                            .with_skip_sorting(true)
                            .with_symmetric(false)
                            .on(device_exec);
            }
            if (smoother == "SSOR") {
                bjfac = sor::build()
                            .with_skip_sorting(true)
                            .with_symmetric(true)
                            .on(device_exec);
            }

            auto single_it = it::build().with_max_iters(1u);
            auto coarse_solve_it = gko::stop::Iteration::build().with_max_iters(
                static_cast<gko::uint32>(maxIterCoarseS));
            auto coarse_solve_norm =
                gko::stop::ResidualNorm<scalar>::build().with_reduction_factor(
                    solveNorm);
            auto smoother_it = gko::stop::Iteration::build().with_max_iters(
                static_cast<gko::uint32>(maxIterS));

            auto smoother_gen =
                type == "Distributed"
                    ? gko::share(ir::build()
                                     .with_solver(
                                         ras::build().with_local_solver(bjfac))
                                     .with_relaxation_factor(relaxFac)
                                     .with_criteria(smoother_it)
                                     .on(device_exec))
                    : gko::share(ir::build()
                                     .with_solver(bjfac)
                                     .with_relaxation_factor(relaxFac)
                                     .with_criteria(smoother_it)
                                     .on(device_exec));

            if (type == "Schwarz") {
                auto pre_factory =
                    mg::build()
                        .with_max_levels(static_cast<gko::uint32>(maxLevels))
                        .with_cycle(cycle)
                        .with_min_coarse_rows(
                            static_cast<gko::uint32>(minRowsC))
                        .with_pre_smoother(smoother_gen)
                        .with_post_uses_pre(true)
                        .with_mg_level(
                            pgm::build().with_deterministic(false).on(
                                device_exec))
                        .with_coarsest_solver(
                            gko::share(cg::build()
                                           .with_preconditioner(bjfac)
                                           .with_criteria(coarse_solve_it,
                                                          coarse_solve_norm)
                                           .on(device_exec)))
                        .with_criteria(single_it)
                        .on(device_exec);
                return wrap_schwarz(gkomatrix, device_exec,
                                    std::move(pre_factory));
            }

            if (type == "Distributed") {
                auto coarsest_gen = gko::share(
                    cg::build()
                        .with_preconditioner(
                            ras::build().with_local_solver(bjfac))
                        .with_criteria(coarse_solve_it, coarse_solve_norm)
                        .on(device_exec));
                auto gkodistmatrix =
                    gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
                auto smoother_gen = gko::share(
                    ir::build()
                        .with_solver(ras::build().with_local_solver(bjfac))
                        .with_relaxation_factor(relaxFac)
                        .with_criteria(smoother_it)
                        .on(device_exec));
                auto ret = gko::share(
                    gko::solver::Multigrid::build()
                        .with_max_levels(maxLevels)
                        .with_mg_level(gko::multigrid::Pgm<scalar>::build()
                                           .with_deterministic(true))
                        .with_min_coarse_rows(minRowsC)
                        .with_coarsest_solver(coarsest_gen)
                        .with_criteria(it::build().with_max_iters(2u))
                        .with_smoother_iters(maxIterS)
                        .with_pre_smoother(smoother_gen)
                        .with_post_uses_pre(true)
                        .with_cycle(cycle)
                        .on(device_exec)
                        ->generate(gkodistmatrix));
                return ret;
            }
            FatalErrorInFunction << "Unknown Multigrid type: " << type
                                 << "\nValid Choices: Schwarz, Distributed"
                                 << abort(FatalError);
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
        std::shared_ptr<const gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec) const
    {
        const word precond_store_name =
            sys_matrix_name_ + "Cached_preconditinoner";
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

        const dictionary &d = e.isDict() ? e.dict() : dictionary::null;

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
                    word msg = "Update Multigrid preconditioner";
                    MLOG_1(verbose_, msg)
                    auto gkodistmatrix =
                        gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
                    label rows = gko::as<gko::experimental::distributed::Matrix<
                        scalar, label, label>>(gkodistmatrix)
                                     ->get_local_matrix()
                                     ->get_size()[0];
                    if (rows == 0) return ret;

                    word type = d.lookupOrDefault("type", word("Schwarz"));

                    if (type == "Schwarz") {
                        auto local_solver = std::const_pointer_cast<gko::LinOp>(
                            gko::as<ras>(ret)->get_local_solver());

                        gko::as<gko::UpdateMatrixValue>(local_solver)
                            ->update_matrix_value(
                                gko::as<gko::experimental::distributed::Matrix<
                                    scalar, label, label>>(gkodistmatrix)
                                    ->get_local_matrix());
                    } else {
                        gko::as<gko::UpdateMatrixValue>(ret)
                            ->update_matrix_value(gkodistmatrix);
                    }
                }
                return ret;
            } else {
                auto prev_precond = db_.template lookupObjectRef<
                    DevicePersistentBase<gko::LinOp>>(precond_store_name);
                const label caching_period =
                    d.lookupOrDefault<label>("caching", 0);
                set_next_caching(sys_matrix_name_, db_, caching_period);

                auto generated_precond =
                    init_preconditioner_impl(name, d, gkomatrix, device_exec);

                auto precond_ptr = prev_precond.get_ptr();
                precond_ptr = generated_precond;
                return precond_ptr;
            }
        }
        const label caching_period = d.lookupOrDefault<label>("caching", 0);
        set_next_caching(sys_matrix_name_, db_, caching_period);
        cache = get_next_caching(sys_matrix_name_, db_);

        auto generated_precond =
            init_preconditioner_impl(name, d, gkomatrix, device_exec);

        auto po = new DevicePersistentBase<gko::LinOp>(IOobject(path, db_),
                                                       generated_precond);

        // use get_ptr(() to avoid unused variable warning
        po->get_ptr();

        return generated_precond;
    }
};
}  // namespace Foam
