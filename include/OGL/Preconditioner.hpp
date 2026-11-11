// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "OGL/DevicePersistent/Base.hpp"
#include "OGL/MatrixWrapper/Distributed.hpp"
#include "OGL/Preconditioner/Adaptive.hpp"
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
        if (name == "BJ") {
            return BlockJacobi(device_exec, gkomatrix, d, verbose_).create();
        }
        if (name == "Multigrid") {
            return Multigrid(device_exec, gkomatrix, d, verbose_)
                .create(db_, sys_matrix_name_);
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

    /* returns the caching period to use for a newly generated
     * preconditioner. While fewer than cachingDelay preconditioners have been
     * generated, caching is disabled, i.e. the preconditioner is regenerated
     * on every solve.
     */
    label caching_period(const dictionary &d) const
    {
        const label period = d.lookupOrDefault<label>("caching", 0);
        const label caching_delay = d.lookupOrDefault<label>("cachingDelay", 0);
        const label generations =
            get_preconditioner_generations(sys_matrix_name_, db_);

        if (generations < caching_delay) {
            set_preconditioner_generations(sys_matrix_name_, db_,
                                           generations + 1);
            word msg = "Postpone preconditioner caching, generation " +
                       std::to_string(generations + 1) + " of " +
                       std::to_string(caching_delay);
            LOG_1(verbose_, msg)
            return 0;
        }
        return period;
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

        set_preconditioner_from_cache(sys_matrix_name_, db_,
                                      stored && cache > 0);

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

                return ret;
            } else {
                auto &prev_precond = db_.template lookupObjectRef<
                    DevicePersistentBase<gko::LinOp>>(precond_store_name);
                set_next_caching(sys_matrix_name_, db_, caching_period(d));

                auto generated_precond =
                    init_preconditioner_impl(name, d, gkomatrix, device_exec);

                // store the new preconditioner for the next caching period
                prev_precond.set_ptr(generated_precond);
                return generated_precond;
            }
        }
        set_next_caching(sys_matrix_name_, db_, caching_period(d));
        cache = get_next_caching(sys_matrix_name_, db_);

        auto generated_precond =
            init_preconditioner_impl(name, d, gkomatrix, device_exec);

        auto po = new DevicePersistentBase<gko::LinOp>(IOobject(path, db_),
                                                       generated_precond);

        // use get_ptr(() to avoid unused variable warning
        po->get_ptr();

        return generated_precond;
    }

    /* Discards the cached preconditioner if the solve using it did not
     * converge, i.e. the next solve generates a new preconditioner. A cached
     * preconditioner can stop fitting the matrix, since the matrix values
     * change between solves.
     *
     * @param converged  whether the solve converged, has to be the same on
     * all ranks to keep the caching state consistent
     */
    void check_cached_preconditioner(bool converged) const
    {
        if (converged ||
            !get_preconditioner_from_cache(sys_matrix_name_, db_)) {
            return;
        }
        word msg =
            "Solve with cached preconditioner did not converge, "
            "regenerate the preconditioner for the next solve";
        MLOG_0(verbose_, msg)
        set_next_caching(sys_matrix_name_, db_, 0);
    }
};


}  // namespace Foam
