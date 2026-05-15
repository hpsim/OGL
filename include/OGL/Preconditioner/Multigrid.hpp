// SPDX-FileCopyrightText: 2025 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "OGL/Preconditioner/CoarseSolver.hpp"
#include "OGL/Preconditioner/Schwarz.hpp"

class Multigrid {
    using ir = gko::solver::Ir<scalar>;
    using it = gko::stop::Iteration;
    using ras =
        gko::experimental::distributed::preconditioner::Schwarz<scalar, label,
                                                                label>;
    using sor = gko::preconditioner::Sor<scalar, label>;
    using dbj = gko::preconditioner::Jacobi<double, label>;
    using fbj = gko::preconditioner::Jacobi<float, label>;
    using mg = gko::solver::Multigrid;
    using pgm = gko::multigrid::Pgm<scalar, label>;

    std::shared_ptr<gko::Executor> exec_;

    std::shared_ptr<const gko::LinOp> mtx_;

    const dictionary &d_;

    const label verbose_;

    bool skip_sorting_;

    bool multi_level_schwarz_;

    word type_;
    scalar relaxFac_;
    word cycleName_;
    label maxLevels_;
    label minRowsC_;
    word smoother_;
    word coarsening_;
    label maxIterS_;

public:
    Multigrid(std::shared_ptr<gko::Executor> exec,
              std::shared_ptr<const gko::LinOp> mtx, const dictionary &d,
              label verbose)
        : exec_(exec),
          mtx_(mtx),
          d_(d),
          verbose_(verbose),
          skip_sorting_(d.lookupOrDefault<Switch>("skipSorting", true)),
          multi_level_schwarz_(
              d.lookupOrDefault<Switch>("multiLevelSchwarz", false)),
          type_(d.lookupOrDefault("type", word("Schwarz"))),
          relaxFac_(d.lookupOrDefault("relaxationFactor", scalar(0.9))),
          cycleName_(d.lookupOrDefault("cycle", word("v"))),
          maxLevels_(d.lookupOrDefault("maxLevels", label(20))),
          minRowsC_(d.lookupOrDefault("minCoarseRows", label(64000))),
          smoother_(d.lookupOrDefault("smoother", word("Jacobi"))),
          coarsening_(d.lookupOrDefault("coarsening", word("PGM"))),
          maxIterS_(d.lookupOrDefault("maxIterSmoother", label(1)))
    {
        word msg = "\nGenerate Multigrid preconditioner:\n\tmaxLevels:" +
                   std::to_string(maxLevels_) +
                   "\n\tminCoarseRows: " + std::to_string(minRowsC_) +
                   "\n\tSmoother: " + smoother_ +
                   "\n\trelaxationFactor: " + std::to_string(relaxFac_) +
                   "\n\tmaxIterSmoother: " + std::to_string(maxIterS_) +
                   "\n\tcoarsening: " + coarsening_ +
                   "\n\tcycle: " + cycleName_ + "\n\ttype: " + type_;
        MLOG_0(verbose_, msg)
        // FatalErrorInFunction << "Unknown Multigrid type: " << type
        //                      << "\nValid Choices: Schwarz, Distributed"
        //                      << abort(FatalError);
    }

    std::shared_ptr<gko::LinOp> create(
        /* const objectRegistry &db,
          const ExecutorHandler &exec_handler*/
    )
    {
        gko::solver::multigrid::cycle cycle;
        if (cycleName_ == "v") cycle = gko::solver::multigrid::cycle::v;
        if (cycleName_ == "w") cycle = gko::solver::multigrid::cycle::w;
        if (cycleName_ == "f") cycle = gko::solver::multigrid::cycle::f;

        std::shared_ptr<gko::LinOpFactory> smootherFac{};
        if (smoother_ == "Jacobi") {
            smootherFac =
                dbj::build().with_max_block_size(1u).with_skip_sorting(true).on(
                    exec_);
        }
        if (smoother_ == "SOR") {
            smootherFac =
                sor::build().with_skip_sorting(true).with_symmetric(false).on(
                    exec_);
        }
        if (smoother_ == "SSOR") {
            smootherFac =
                sor::build().with_skip_sorting(true).with_symmetric(true).on(
                    exec_);
        }
        if (smootherFac == nullptr) {
            FatalErrorInFunction << "Unknown smoother: " << smoother_
                                 << "\nValid Choices: Jacobi, SOR, SSOR"
                                 << abort(FatalError);
        }

        // std::shared_ptr<gko::matrix::Csr<scalar, label>> coarseningWeight;
        // if (coarsening_ == "GAMG") {
        // }

        // if (coarsening_ == "PGM") {
        //     // auto repartDistMtx = gko::as<RepartDistMatrix>(mtx_);
        //     // auto cw = gko::as<gko::matrix::Csr<scalar, label>>(
        //     //     repartDistMtx->get_local_matrix());
        //     coarseningWeight = nullptr;
        //     // std::const_pointer_cast<gko::matrix::Csr<scalar, label>>(cw);
        // }
        // // if (coarseningWeight == nullptr) {
        // //     FatalErrorInFunction << "Unknown coarsening: " << coarsening_
        // //                          << "\nValid Choices: GAMG, PGM"
        // //                          << abort(FatalError);
        // // }

        auto single_it = it::build().with_max_iters(1u);
        auto smoother_it = gko::stop::Iteration::build().with_max_iters(
            static_cast<gko::uint32>(maxIterS_));

        auto smoother_gen =
            type_ == "Distributed"
                ? gko::share(ir::build()
                                 .with_solver(ras::build().with_local_solver(
                                     smootherFac))
                                 .with_relaxation_factor(relaxFac_)
                                 .with_criteria(smoother_it)
                                 .on(exec_))
                : gko::share(ir::build()
                                 .with_solver(smootherFac)
                                 .with_relaxation_factor(relaxFac_)
                                 .with_criteria(smoother_it)
                                 .on(exec_));

        if (type_ == "Schwarz") {
            auto pre_factory = gko::share(
                mg::build()
                    .with_max_levels(static_cast<gko::uint32>(maxLevels_))
                    .with_cycle(cycle)
                    .with_min_coarse_rows(static_cast<gko::uint32>(minRowsC_))
                    .with_pre_smoother(smoother_gen)
                    .with_smoother_iters(maxIterS_)
                    .with_post_uses_pre(true)
                    .with_mg_level(
                        pgm::build()
                            .with_deterministic(true)
                            // .with_local_weight_mtx(coarseningWeight)
                            .on(exec_))
                    .with_coarsest_solver(
                        generate_coarse_solver(exec_, d_, verbose_, false))
                    .with_criteria(single_it)
                    .on(exec_));
            return wrap_schwarz(
                mtx_, exec_,
                gko::share(pre_factory->generate(
                    gko::as<RepartDistMatrix>(mtx_)->get_local())));
        }

        if (type_ == "Distributed") {
            // std::shared_ptr<const gko::LinOpFactory> coarsest_solver{};
            // NOTE does not support distributed currently
            // if (coarseSolver_ == "Direct") {
            //     coarsest_solver = gko::share(
            //         gko::experimental::solver::Direct<scalar, label>::build()
            //             .on(exec_));
            // }
            auto gkodistmatrix =
                gko::as<RepartDistMatrix>(mtx_)->get_dist_matrix();
            auto smoother_gen = gko::share(
                ir::build()
                    .with_solver(ras::build().with_local_solver(smootherFac))
                    .with_relaxation_factor(relaxFac_)
                    .with_criteria(smoother_it)
                    .on(exec_));
            auto ret = gko::share(
                mg::build()
                    .with_max_levels(maxLevels_)
                    .with_cycle(cycle)
                    .with_min_coarse_rows(minRowsC_)
                    .with_pre_smoother(smoother_gen)
                    .with_smoother_iters(maxIterS_)
                    .with_post_uses_pre(true)
                    .with_mg_level(
                        pgm::build()
                            // .with_local_weight_mtx(coarseningWeight)
                            .with_deterministic(true))
                    .with_coarsest_solver(
                        generate_coarse_solver(exec_, d_, verbose_))
                    .with_criteria(single_it)
                    .on(exec_)
                    ->generate(gkodistmatrix));
            return ret;
        }

        return {};
    }
};
