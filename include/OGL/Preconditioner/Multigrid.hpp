// SPDX-FileCopyrightText: 2025 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "OGL/DevicePersistent/Base.hpp"
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
    bool reuseHierarchy_;

    /* Generate the multigrid from its factory. With reuseHierarchy the
     * hierarchy recorded by the first call is kept in the registry and reused
     * by later calls, which then only recompute the coarse matrices, smoothers
     * and the coarsest solver. If the matrix no longer fits the recorded
     * hierarchy, e.g. after a change of the sparsity pattern, the recording
     * starts again.
     *
     * NOTE Ginkgo requires a result of generate_reuse not to outlive its reuse
     * data. The multigrid hierarchy and its Pgm levels are independent of the
     * reuse data and the other components do not support reuse, so the stored
     * preconditioner does not depend on the lifetime of the reuse data.
     */
    std::shared_ptr<gko::LinOp> generate(
        std::shared_ptr<const gko::LinOpFactory> factory,
        std::shared_ptr<const gko::LinOp> mtx, const objectRegistry &db,
        const word &store_name) const
    {
        if (!reuseHierarchy_) {
            return gko::share(factory->generate(mtx));
        }

        using reuse_data = gko::LinOpFactory::ReuseData;
        const word reuse_store_name = store_name + "_multigrid_reuse_data";
        if (!db.foundObject<regIOobject>(reuse_store_name)) {
            word msg = "Record Multigrid hierarchy for reuse";
            MLOG_0(verbose_, msg)
            // registers itself in the object registry
            new DevicePersistentBase<reuse_data>(
                IOobject(reuse_store_name, db),
                gko::share(factory->create_empty_reuse_data()));
        } else {
            word msg = "Reuse recorded Multigrid hierarchy";
            MLOG_1(verbose_, msg)
        }
        auto &stored = db.lookupObjectRef<DevicePersistentBase<reuse_data>>(
            reuse_store_name);

        try {
            return gko::share(factory->generate_reuse(mtx, *stored.get_ptr()));
        } catch (const gko::Error &e) {
            word msg = word(
                           "Recorded Multigrid hierarchy does not fit the "
                           "matrix, recording again: ") +
                       e.what();
            MLOG_0(verbose_, msg)
            stored.set_ptr(gko::share(factory->create_empty_reuse_data()));
            return gko::share(factory->generate_reuse(mtx, *stored.get_ptr()));
        }
    }

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
          maxIterS_(d.lookupOrDefault("maxIterSmoother", label(1))),
          reuseHierarchy_(d.lookupOrDefault<Switch>("reuseHierarchy", false))
    {
        word msg = "\nGenerate Multigrid preconditioner:\n\ttype: " + type_ +
                   "\n\tmaxLevels: " + std::to_string(maxLevels_) +
                   "\n\tminCoarseRows: " + std::to_string(minRowsC_) +
                   "\n\tsmoother: " + smoother_ +
                   "\n\trelaxationFactor: " + std::to_string(relaxFac_) +
                   "\n\tmaxIterSmoother: " + std::to_string(maxIterS_) +
                   "\n\tcoarsening: " + coarsening_ +
                   "\n\tcycle: " + cycleName_ +
                   "\n\treuseHierarchy: " + Switch(reuseHierarchy_).c_str();
        MLOG_0(verbose_, msg)
    }

    /* @param db  registry to keep the reuse data of the hierarchy in
     * @param store_name  name of the system matrix the preconditioner is for
     */
    std::shared_ptr<gko::LinOp> create(const objectRegistry &db,
                                       const word &store_name)
    {
        gko::solver::multigrid::cycle cycle;
        if (cycleName_ == "v") {
            cycle = gko::solver::multigrid::cycle::v;
        } else if (cycleName_ == "w") {
            cycle = gko::solver::multigrid::cycle::w;
        } else if (cycleName_ == "f") {
            cycle = gko::solver::multigrid::cycle::f;
        } else {
            FatalErrorInFunction << "Unknown cycle: " << cycleName_
                                 << "\nValid Choices: v, w, f"
                                 << abort(FatalError);
        }

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
                generate(pre_factory,
                         gko::as<RepartDistMatrix>(mtx_)->get_local(), db,
                         store_name));
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
            auto mg_factory = gko::share(
                mg::build()
                    .with_max_levels(static_cast<gko::uint32>(maxLevels_))
                    .with_cycle(cycle)
                    .with_min_coarse_rows(static_cast<gko::uint32>(minRowsC_))
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
                    .on(exec_));
            return generate(mg_factory, gkodistmatrix, db, store_name);
        }

        FatalErrorInFunction << "Unknown Multigrid type: " << type_
                             << "\nValid Choices: Schwarz, Distributed"
                             << abort(FatalError);
        return {};
    }
};
