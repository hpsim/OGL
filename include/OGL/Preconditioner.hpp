// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "OGL/DevicePersistent/Base.hpp"
#include "OGL/MatrixWrapper/Distributed.hpp"
#include "OGL/Preconditioner/Cholesky.hpp"
#include "OGL/Preconditioner/ISAI.hpp"
#include "OGL/Preconditioner/Jacobi.hpp"
#include "OGL/Preconditioner/LU.hpp"
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


    std::shared_ptr<gko::LinOp> init_preconditioner_impl(
        const word name, const dictionary &d,
        std::shared_ptr<const gko::LinOp> gkomatrix,
        std::shared_ptr<gko::Executor> device_exec) const
    {
        bool skip_sorting = d.lookupOrDefault<Switch>("skipSorting", true);
        bool multi_level_schwarz =
            d.lookupOrDefault<Switch>("multiLevelSchwarz", false);

        if (name == "BJ") {
            return BlockJacobi(device_exec, gkomatrix, d, verbose_).create();
        }
        if (name == "Multigrid") {
            return Multigrid(device_exec, gkomatrix, d, verbose_)
                .create(/*db_, exec_handler*/);
        }
        if (name == "ILU") {
            return LU(device_exec, gkomatrix, d, verbose_).create();
        }
        if (name == "IC") {
            return Cholesky(device_exec, gkomatrix, d, verbose_).create();
        }
        if (name == "ISAI") {
            return ISAI(device_exec, gkomatrix, d, verbose_).create();
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

#ifdef GINKGO_WITH_OGL_EXTENSION
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
#endif
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
