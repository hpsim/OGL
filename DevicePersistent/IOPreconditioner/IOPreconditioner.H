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

#include "../DevicePersistentBase/DevicePersistentBase.H"


namespace Foam {
struct PreconditionerInitFunctor {
    const PersistentArray<label> &col_idxs_;
    const PersistentArray<label> &row_idxs_;
    const PersistentArray<scalar> &values_;
    const label verbose_;

    PreconditionerInitFunctor(const PersistentArray<label> &col_idxs,
                              const PersistentArray<label> &row_idxs,
                              const PersistentArray<scalar> &values,
                              const label verbose)
        : col_idxs_(col_idxs),
          row_idxs_(row_idxs),
          values_(values),
          verbose_(verbose)
    {}


    void update(std::shared_ptr<mtx> &csr_matrix) const
    {
        auto values_view = val_array::view(values_.get_device_exec(),
                                           values_.get_global_size(),
                                           csr_matrix->get_values());

        // copy values to device
        values_view = *values_.get_array().get();
    }

    std::shared_ptr<mtx> init() const
    {
        const auto device_exec = values_.get_device_exec();
        const label nCells = values_.get_global_size();

        auto coo_mtx = gko::share(coo_mtx::create(
            device_exec, gko::dim<2>(nCells, nCells),
            val_array(device_exec, *values_.get_array().get()),
            *col_idxs_.get_array().get(), *row_idxs_.get_array().get()));

        auto gkomatrix =
            gko::share(mtx::create(device_exec, gko::dim<2>(nCells, nCells)));

        SIMPLE_TIME(verbose_, convert_coo_to_csr,
                    coo_mtx->convert_to(gkomatrix.get());)
        return gkomatrix;
    }
};

class IOPreconditioner {
    using bj = gko::preconditioner::Jacobi<>;
    using ic = gko::preconditioner::Ic<>;

private:
    const word name_;

    const label maxBlockSize_;

    const word preconditioner_db_name_;

    const bool preconditioner_stored_;

    const bool update_preconditioner_;

    mutable PersistentBase<, class InitFunctor> preconditioner_;
    // mutable GKOLinOPIOPtr *io_precond_ptr_ = NULL;

public:
    IOPreconditioner(const objectRegistry &db, const dictionary &solverControls,
                     const word fieldName)
        : name_(solverControls.lookupOrDefault("preconditioner", word("none"))),
          maxBlockSize_(
              solverControls.lookupOrDefault("maxBlockSize", label(16))),
          preconditioner_db_name_("preconditioner_" + fieldName),
          preconditioner_stored_(
              db.foundObject<regIOobject>(preconditioner_db_name_)),
          update_preconditioner_(solverControls.lookupOrDefault<Switch>(
              "update_preconditioner", true)){};

    void init_preconditioner(const objectRegistry &db,
                             std::shared_ptr<mtx> gkomatrix,
                             std::shared_ptr<gko::Executor> device_exec) const
    {
        // TODO get field specific preconditioner
        if (preconditioner_stored_ && !update_preconditioner_) {
            io_precond_ptr_ =
                &db.lookupObjectRef<GKOLinOPIOPtr>(preconditioner_db_name_);
            return;
        }
        const fileName path = preconditioner_db_name_;
        if (name_ == "BJ") {
            // TODO for non constant system matrix reuse block pointers
            auto pre_factory = bj::build()
                                   .with_skip_sorting(true)
                                   .with_max_block_size(maxBlockSize_)
                                   .on(device_exec);
            auto pre_generated = gko::share(pre_factory->generate(gkomatrix));
            io_precond_ptr_ =
                new GKOLinOPIOPtr(IOobject(path, db), pre_generated);
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
            auto pre_generated =
                gko::share(ilu_pre_factory->generate(gko::share(par_ilu)));

            io_precond_ptr_ =
                new GKOLinOPIOPtr(IOobject(path, db), pre_generated);
        }
        if (name_ == "IC") {
            auto factorizaton_factory =
                gko::factorization::Ic<scalar, label>::build()
                    .with_skip_sorting(true)
                    .on(device_exec);

            auto ic = gko::share(factorizaton_factory->generate(gkomatrix));
            auto pre_factory =
                gko::preconditioner::Ic<>::build().on(device_exec);

            auto pre_generated = gko::share(pre_factory->generate(ic));
            io_precond_ptr_ =
                new GKOLinOPIOPtr(IOobject(path, db), pre_generated);
        }
        if (name_ == "ISAI") {
            auto pre_factory =
                gko::preconditioner::Isai<gko::preconditioner::isai_type::spd,
                                          scalar, label>::build()
                    .with_skip_sorting(true)
                    .on(device_exec);

            auto pre_generated = gko::share(pre_factory->generate(gkomatrix));
            io_precond_ptr_ =
                new GKOLinOPIOPtr(IOobject(path, db), pre_generated);
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

            auto pre_generated = gko::share(pre_factory->generate(gkomatrix));
            io_precond_ptr_ =
                new GKOLinOPIOPtr(IOobject(path, db), pre_generated);
        }
    };

    std::shared_ptr<gko::LinOp> get_preconditioner() const
    {
        if (io_precond_ptr_ == NULL) {
            return std::shared_ptr<gko::LinOp>{};
        }
        return io_precond_ptr_->get_ptr();
    }
};
}  // namespace Foam

#endif
