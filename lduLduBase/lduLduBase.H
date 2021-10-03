/*---------------------------------------------------------------------------*\
License
    This file is part of OGL.

    OGL is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OGL.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::IOSortingIdxHandler

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    lduLduBase.H

\*---------------------------------------------------------------------------*/

#ifndef OGL_lduLduBase_INCLUDED_H
#define OGL_lduLduBase_INCLUDED_H

namespace Foam {

#include "../IOExecutorHandler/IOExecutorHandler.H"
#include "../IOGKOMatrixHandler/IOGKOMatrixHandler.H"
#include "../IOHandler/IOPreconditioner/IOPreconditioner.H"
#include "../IOHandler/IOSortingIdxHandler/IOSortingIdxHandler.H"
#include "../common/StoppingCriterion.H"
#include "../common/common.H"

#include <ginkgo/ginkgo.hpp>
#include <map>

template <class MatrixType, class SolverFactory>
class lduLduBase : public HostMatrix<MatrixType>,
                   public SolverFactory,
                   public StoppingCriterion,
                   public IOSortingIdxHandler,
                   public IOGKOMatrixHandler,
                   public IOPreconditioner {
private:
    const bool verbose_;

public:
    lduLduBase(const word &fieldName, const lduMatrix &matrix,
               const FieldField<Field, scalar> &interfaceBouCoeffs,
               const FieldField<Field, scalar> &interfaceIntCoeffs,
               const lduInterfaceFieldPtrsList &interfaces,
               const dictionary &solverControls)
        : HostMatrix<lduMatrix>(fieldName, matrix, interfaceBouCoeffs,
                                interfaceIntCoeffs, interfaces, solverControls),
          SolverFactory(solverControls),
          StoppingCriterion(solverControls),
          IOSortingIdxHandler(
              matrix.mesh().thisDb(), this->nElems(),
              solverControls.lookupOrDefault<Switch>("sort", true)),
          IOGKOMatrixHandler(matrix.mesh().thisDb(), solverControls, fieldName),
          IOPreconditioner(matrix.mesh().thisDb(), solverControls, fieldName),
          verbose_(solverControls.lookupOrDefault<Switch>("verbose", true))
    {
        init_base();
    }


    //- Construct from matrix components and solver controls
    lduLduBase(const word &fieldName, const MatrixType &matrix,
               const dictionary &solverControls)
        : HostMatrix<MatrixType>(fieldName, matrix, solverControls),
          SolverFactory(solverControls),
          StoppingCriterion(solverControls),
          IOSortingIdxHandler(
              matrix.mesh().thisDb(), this->nElems(),
              solverControls.lookupOrDefault<Switch>("sort", true)),
          IOGKOMatrixHandler(matrix.mesh().thisDb(), solverControls, fieldName),
          IOPreconditioner(matrix.mesh().thisDb(), solverControls, fieldName),
          verbose_(solverControls.lookupOrDefault<Switch>("verbose", false))
    {
        init_base();
    }

    void init_base()
    {
        // if sys_matrix is not stored updating is neccesary
        // initially
        bool stored = get_sys_matrix_stored();
        if (!stored) {
            SIMPLE_TIME(verbose_, init_host_sparsity_pattern,
                        this->init_host_sparsity_pattern();)
            SIMPLE_LOG(verbose_, "matrix not stored update host matrix")
            SIMPLE_TIME(verbose_, update_host_mtx,
                        this->update_host_matrix_data();)
        } else {
            // if sys_matrix is  stored updating is only neccesary
            // when requested explictly
            if (get_update_sys_matrix()) {
                SIMPLE_LOG(
                    verbose_,
                    "matrix is stored and update of host matrix requested")
                SIMPLE_TIME(verbose_, exp_update_host_mtx,
                            this->update_host_matrix_data();)
            }
        }

        // after updating the host matrix the host matrix needs to be sorted
        if (!get_is_sorted()) {
            SIMPLE_LOG(verbose_,
                       "matrix is not yet sorted, compute sorting idxs")
            SIMPLE_TIME(
                verbose_, compute_sort,
                this->compute_sorting_idxs(this->row_idxs(), this->col_idxs(),
                                           this->nCells());)
        }

        if (!stored && get_sort()) {
            SIMPLE_LOG(verbose_,
                       "matrix is not yet sorted, sorting host matrix")
            SIMPLE_TIME(
                verbose_, sort_host_mtx_sparsity_pattern,
                this->sort_host_matrix_sparsity_pattern(get_sorting_idxs());)
            SIMPLE_TIME(verbose_, sort_host_mtx,
                        this->sort_host_matrix_data(this->get_sorting_idxs());)
        }

        init_device_matrix(this->matrix().mesh().thisDb(), this->values(),
                           this->col_idxs(), this->row_idxs(), this->nElems(),
                           this->nCells(), !get_update_sys_matrix());
    }

    scalar compute_normfactor(std::shared_ptr<gko::Executor> device_exec,
                              std::shared_ptr<mtx> gkomatrix,
                              std::shared_ptr<vec> x,
                              std::shared_ptr<vec> b) const
    {
        // SIMPLE_TIME(verbose_, compute_col_sum_A, [=]() {
        // computes  || Ax - x* ||_1 + || b - x* ||
        val_array x_array{
            val_array::view(device_exec, this->nCells(), x->get_values())};

        // x_array.set_executor(ref_exec());

        auto xAvg =
            gko::initialize<gko::matrix::Dense<scalar>>(1, {0.0}, device_exec);

        ::gko::reduce(x_array,
                      xAvg->get_values());  // / this->nCells()}, device_exec);

        auto one_over_nCells = gko::initialize<gko::matrix::Dense<scalar>>(
            1, {1.0 / this->nCells()}, device_exec);

        xAvg->scale(one_over_nCells.get());
        auto xAvg_host =
            vec::create(device_exec->get_master(), gko::dim<2>{1, 1});
        xAvg_host->copy_from(xAvg.get());

        // TODO store colA vector
        auto colA = gko::share(
            vec::create(device_exec, gko::dim<2>{this->nCells(), 1}));
        //
        // gkomatrix->compute_column_vector_sum(colA.get());

        // if column vector sum is not availible use dot product
        auto xAvg_vec = gko::share(
            vec::create(device_exec, gko::dim<2>{this->nCells(), 1}));

        auto unity =
            gko::initialize<gko::matrix::Dense<scalar>>(1, {1.0}, device_exec);

        xAvg_vec->fill(xAvg_host->get_values()[0]);
        gkomatrix->apply(xAvg_vec.get(), colA.get());

        auto b_clone = b->clone();
        b_clone->sub_scaled(unity.get(), colA.get());
        b_clone->compute_absolute_inplace();


        auto Ax = gko::share(
            vec::create(device_exec, gko::dim<2>{this->nCells(), 1}));
        gkomatrix->apply(x.get(), Ax.get());

        Ax->sub_scaled(unity.get(), colA.get());

        auto norm_1 = gko::share(vec::create(device_exec, gko::dim<2>{1, 1}));
        Ax->compute_absolute_inplace();

        Ax->add_scaled(unity.get(), b_clone.get());
        Ax->compute_norm1(norm_1.get());


        auto res_host =
            vec::create(device_exec->get_master(), gko::dim<2>{1, 1});
        res_host->copy_from(norm_1.get());
        return res_host->get_values()[0];
    }


    template <class OFField>
    void init_vector_views(std::vector<val_array> &target,
                           OFField &source) const
    {
        for (int i = 0; i < 3; i++) {
            target.push_back(
                val_array::view(ref_exec(), 3 * this->nCells(),
                                const_cast<scalar *>(&source[0][i])));
        }
    }

    template <class Type>
    SolverPerformance<Type> solve_impl_(Field<Type> &psi) const
    {
        std::shared_ptr<gko::Executor> device_exec =
            this->get_device_executor();

        // --- Setup class containing solver performance data
        // Implement
        word preconditionerName(this->controlDict_.lookup("preconditioner"));

        SolverPerformance<Type> solverPerf(
            this->get_device_executor_name() + preconditionerName,
            this->fieldName_);

        std::vector<val_array> source_views{};

        init_vector_views(source_views, this->matrix().source());

        std::vector<std::shared_ptr<vec>> b{};
        for (int i = 0; i < 3; i++) {
            b.push_back(vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1),
                                    source_views[i], 3));
        }

        init_initial_guess_vector(psi, this->matrix().mesh().thisDb(),
                                  this->nCells());
        std::vector<std::shared_ptr<vec>> x{};
        this->get_initial_guess(x);

        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec{};

        criterion_vec.push_back(build_stopping_criterion(device_exec, 1.0));

        // Generate solver
        auto solver_gen = this->create_solver(device_exec, criterion_vec);

        std::shared_ptr<mtx> gkomatrix = get_gkomatrix();

        auto solver = solver_gen->generate(gko::share(gkomatrix));

        for (int i = 0; i < 3; i++) {
            SIMPLE_TIME(verbose_, solve,
                        solver->apply(gko::lend(b[i]), gko::lend(x[i]));)
        }

        // copy back
        copy_result_back(psi, this->nCells());

        return solverPerf;
    };

    solverPerformance solve_impl_(word typeName, scalarField &psi,
                                  const scalarField &source,
                                  const direction cmpt = 0) const
    {
        std::shared_ptr<gko::Executor> device_exec =
            this->get_device_executor();

        // --- Setup class containing solver performance data
        solverPerformance solverPerf(
            lduMatrix::preconditioner::getName(this->controlDict_) +
                this->get_device_executor_name() + typeName,
            this->fieldName());


        scalarField pA(this->nCells());
        scalarField wA(this->nCells());

        SIMPLE_TIME(verbose_, Amul,
                    this->matrix().Amul(wA, psi, this->interfaceBouCoeffs_,
                                        this->interfaces_, cmpt);)
        scalar norm_factor = 0;


        SIMPLE_TIME(verbose_, compute_of_norm_factor,
                    norm_factor = this->normFactor(psi, source, wA, pA);)

        auto source_view = val_array::view(ref_exec(), this->nCells(),
                                           const_cast<scalar *>(&source[0]));

        std::vector<val_array> source_views{source_view};
        auto b = gko::share(vec::create(
            ref_exec(), gko::dim<2>(this->nCells(), 1), source_views[0], 1));

        init_initial_guess_vector(psi, this->matrix().mesh().thisDb(),
                                  this->nCells());
        std::vector<std::shared_ptr<vec>> x{};
        this->get_initial_guess(x);

        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec{};

        criterion_vec.push_back(
            build_stopping_criterion(device_exec, norm_factor));

        // Generate solver
        std::shared_ptr<mtx> gkomatrix = get_gkomatrix();

        scalar gko_normfactor = 0;

        SIMPLE_TIME(verbose_, compute_gko_norm_factor,
                    gko_normfactor =
                        compute_normfactor(device_exec, gkomatrix, x[0], b);)

        SIMPLE_TIME(verbose_, init_preconditioner,
                    this->init_preconditioner(this->matrix().mesh().thisDb(),
                                              gkomatrix, device_exec);)

        Info << " gko_norm_factor " << gko_normfactor << " of_norm_factor "
             << norm_factor << endl;

        auto solver_gen = this->create_solver(device_exec, criterion_vec,
                                              this->get_preconditioner());

        // Instantiate a ResidualLogger logger.
        auto logger = std::make_shared<IterationLogger>(device_exec);

        // Add the previously created logger to the solver factory. The
        // logger will be automatically propagated to all solvers created
        // from this factory.
        solver_gen->add_logger(logger);

        if (get_export())
            export_system(this->fieldName(), gko::lend(gkomatrix),
                          gko::lend(x[0]), gko::lend(b),
                          this->matrix().mesh().thisDb().time().timeName());


        solverPerf.initialResidual() = this->get_init_res_norm();

        auto solver = solver_gen->generate(gko::share(gkomatrix));

        // Solve system
        SIMPLE_TIME(verbose_, solve,
                    solver->apply(gko::lend(b), gko::lend(x[0]));)

        this->copy_result_back(psi, this->nCells());

        solverPerf.initialResidual() = this->get_init_res_norm();
        solverPerf.finalResidual() = this->get_res_norm();

        solverPerf.nIterations() = logger->get_iters();

        return solverPerf;
    };
};
}  // namespace Foam

#endif
