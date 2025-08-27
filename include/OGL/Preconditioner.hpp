// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "OGL/DevicePersistent/Base.hpp"
#include "OGL/MatrixWrapper/Distributed.hpp"
#include "OGL/Preconditioner/Jacobi.hpp"
#include "OGL/Preconditioner/Multigrid.hpp"
#include "OGL/Preconditioner/Schwarz.hpp"

#include "fvCFD.H"
#include "regIOobject.H"

namespace Foam {


class Preconditioner {
    using mtx = gko::matrix::Csr<scalar>;
    using bj = gko::preconditioner::Jacobi<scalar, label>;
    using fbj = gko::preconditioner::Jacobi<float, label>;
    using dbj = gko::preconditioner::Jacobi<double, label>;
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


    std::shared_ptr<gko::LinOp> init_preconditioner_impl(
        const word name, const dictionary &d,
        std::shared_ptr<const gko::LinOp> gkomatrix,
        const ExecutorHandler &exec_handler) const
    {
        auto device_exec = exec_handler.get_device_exec();
        bool skip_sorting = d.lookupOrDefault<Switch>("skipSorting", true);
        bool multi_level_schwarz =
            d.lookupOrDefault<Switch>("multiLevelSchwarz", false);

        if (name == "BJ") {
            return BlockJacobi(device_exec, gkomatrix, d, verbose_).create();
        }
        if (name == "Multigrid") {
            return Multigrid(device_exec, gkomatrix, d, verbose_)
                .create(db_, exec_handler);
        }
        // if (name == "ILU") {
        //     label iterations(d.lookupOrDefault("iterations", label(0)));
        //     word msg = "Generate preconditioner " + name;
        //     MLOG_0(verbose_, msg)

        //     auto factorization_factory =
        //         gko::factorization::ParIlu<scalar, label>::build()
        //             .with_skip_sorting(skip_sorting)
        //             .with_iterations(iterations)
        //             .on(device_exec);
        //     auto gkodistmatrix =
        //         gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
        //     auto factorization = gko::share(factorization_factory->generate(
        //         gko::as<gko::experimental::distributed::Matrix<
        //             scalar, label, label>>(gkodistmatrix)
        //             ->get_local_matrix()));
        //     auto precond_factory =
        //         gko::preconditioner::Ilu<>::build().on(device_exec);
        //     return wrap_schwarz(gkodistmatrix, device_exec,
        //                         std::move(precond_factory), factorization);
        // }
        // if (name == "ILUT") {
        //     word msg = "Generate preconditioner " + name;
        //     MLOG_0(verbose_, msg)
        //     auto gkodistmatrix =
        //         gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();

        //     auto factorization_factory =
        //         gko::factorization::ParIlut<scalar, label>::build()
        //             .with_skip_sorting(skip_sorting)
        //             .on(device_exec);

        //     auto factorization = gko::share(factorization_factory->generate(
        //         gko::as<gko::experimental::distributed::Matrix<>>(gkodistmatrix)
        //             ->get_local_matrix()));
        //     auto precond_factory =
        //         gko::preconditioner::Ilu<>::build().on(device_exec);
        //     return wrap_schwarz(gkodistmatrix, device_exec,
        //                         std::move(precond_factory), factorization);
        // }
        // if (name == "IRILU") {
        //     auto trisolve_factory =
        //         ir::build()
        //             .with_solver(
        //                 bj::build().with_max_block_size(1u).on(device_exec))
        //             .with_criteria(
        //                 gko::stop::Iteration::build().with_max_iters(5u).on(
        //                     device_exec))
        //             .on(device_exec);

        //     // Generate an ILU preconditioner factory by setting lower and
        //     // upper triangular solver - in this case the previously defined
        //     // iterative refinement method.
        //     auto precond_factory =
        //         gko::preconditioner::Ilu<ir, ir>::build()
        //             .with_l_solver(gko::clone(trisolve_factory))
        //             .with_u_solver(gko::clone(trisolve_factory))
        //             .on(device_exec);

        //     auto factorization_factory =
        //         gko::factorization::Ilu<scalar, label>::build()
        //             .with_skip_sorting(skip_sorting)
        //             .on(device_exec);

        //     auto factorization = gko::share(factorization_factory->generate(
        //         gko::as<gko::experimental::distributed::Matrix<>>(gkomatrix)
        //             ->get_local_matrix()));

        //     // Use incomplete factors to generate ILU preconditioner
        //     return wrap_schwarz(gkomatrix, device_exec,
        //                         std::move(precond_factory), factorization);
        // }
        // if (name == "IC") {
        //     word msg = "Generate preconditioner " + name;
        //     MLOG_0(verbose_, msg)

        //     auto factorization_factory =
        //         gko::factorization::Ic<scalar, label>::build()
        //             .with_skip_sorting(skip_sorting)
        //             .on(device_exec);
        //     auto gkodistmatrix =
        //         gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
        //     auto factorization = gko::share(factorization_factory->generate(
        //         gko::as<gko::experimental::distributed::Matrix<
        //             scalar, label, label>>(gkodistmatrix)
        //             ->get_local_matrix()));
        //     auto precond_factory =
        //         gko::preconditioner::Ic<>::build().on(device_exec);
        //     return wrap_schwarz(gkodistmatrix, device_exec,
        //                         std::move(precond_factory), factorization);
        // }

        // if (name == "ICT") {
        //     bool approx_select(d.lookupOrDefault("approximateSelect", true));
        //     word msg = "Generate preconditioner " + name +
        //                " with approximate select " +
        //                std::to_string(approx_select);
        //     MLOG_0(verbose_, msg)

        //     auto factorization_factory =
        //         gko::factorization::ParIct<scalar, label>::build()
        //             .with_skip_sorting(skip_sorting)
        //             .on(device_exec);

        //     auto gkodistmatrix =
        //         gko::as<RepartDistMatrix>(gkomatrix)->get_dist_matrix();
        //     auto ic_factorization =
        //     gko::share(factorization_factory->generate(
        //         gko::as<gko::experimental::distributed::Matrix<
        //             scalar, label, label>>(gkodistmatrix)
        //             ->get_local_matrix()));

        //     auto precond_factory =
        //         gko::preconditioner::Ic<>::build().on(device_exec);

        //     return wrap_schwarz(gkomatrix, device_exec,
        //                         std::move(precond_factory),
        //                         ic_factorization);
        // }
        // if (name == "ISAI") {
        //     label sparsity_power(d.lookupOrDefault("sparsityPower",
        //     label(1)));

        //     word msg = "Generate preconditioner " + name + " SparsityPower "
        //     +
        //                std::to_string(sparsity_power);
        //     MLOG_0(verbose_, msg)

        //     auto pre_factory =
        //         gko::preconditioner::Isai<gko::preconditioner::isai_type::spd,
        //                                   scalar, label>::build()
        //             .with_skip_sorting(skip_sorting)
        //             .with_sparsity_power(sparsity_power)
        //             .on(device_exec);

        //     return wrap_schwarz(gkomatrix, device_exec,
        //     std::move(pre_factory));
        // }
        // if (name == "GISAI") {
        //     label sparsity_power(d.lookupOrDefault("sparsityPower",
        //     label(1)));

        //     word msg = "Generate preconditioner " + name + " SparsityPower "
        //     +
        //                std::to_string(sparsity_power);
        //     MLOG_0(verbose_, msg)

        //     auto pre_factory = gko::preconditioner::Isai<
        //                            gko::preconditioner::isai_type::general,
        //                            scalar, label>::build()
        //                            .with_skip_sorting(skip_sorting)
        //                            .with_sparsity_power(sparsity_power)
        //                            .on(device_exec);

        //     return wrap_schwarz(gkomatrix, device_exec,
        //     std::move(pre_factory));
        // }
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
        const ExecutorHandler &exec_handler) const
    {
        auto device_exec = exec_handler.get_device_exec();
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
                    init_preconditioner_impl(name, d, gkomatrix, exec_handler);

                auto precond_ptr = prev_precond.get_ptr();
                precond_ptr = generated_precond;
                return precond_ptr;
            }
        }
        const label caching_period = d.lookupOrDefault<label>("caching", 0);
        set_next_caching(sys_matrix_name_, db_, caching_period);
        cache = get_next_caching(sys_matrix_name_, db_);

        auto generated_precond =
            init_preconditioner_impl(name, d, gkomatrix, exec_handler);

        auto po = new DevicePersistentBase<gko::LinOp>(IOobject(path, db_),
                                                       generated_precond);

        // use get_ptr(() to avoid unused variable warning
        po->get_ptr();

        return generated_precond;
    }
};


}  // namespace Foam
