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
#include "../common/common.H"

#include "../IOExecutorHandler/IOExecutorHandler.H"
#include "../IOGKOMatrixHandler/IOGKOMatrixHandler.H"
#include "../IOHandler/IOPreconditioner/IOPreconditioner.H"
#include "../IOHandler/IOSortingIdxHandler/IOSortingIdxHandler.H"
#include "../common/StoppingCriterion.H"

#include <ginkgo/ginkgo.hpp>
#include <map>

template <class MatrixType, class SolverFactory>
class lduLduBase : public HostMatrix<MatrixType>,
                   public SolverFactory,
                   public IOSortingIdxHandler,
                   public IOGKOMatrixHandler,
                   public IOPreconditioner {
private:
    using vec = gko::matrix::Dense<scalar>;
    const dictionary &solverControls_;
    const bool verbose_;

public:
    lduLduBase(const word &fieldName, const lduMatrix &matrix,
               const FieldField<Field, scalar> &interfaceBouCoeffs,
               const FieldField<Field, scalar> &interfaceIntCoeffs,
               const lduInterfaceFieldPtrsList &interfaces,
               const dictionary &solverControls)
        : HostMatrix<lduMatrix>(fieldName, matrix, interfaceBouCoeffs,
                                interfaceIntCoeffs, interfaces, solverControls),
          SolverFactory(solverControls, matrix.mesh().thisDb(), fieldName),
          IOSortingIdxHandler(
              matrix.mesh().thisDb(), this->nElems(),
              solverControls.lookupOrDefault<Switch>("sort", true)),
          IOGKOMatrixHandler(matrix.mesh().thisDb(), solverControls, fieldName),
          IOPreconditioner(matrix.mesh().thisDb(), solverControls, fieldName),
          solverControls_(solverControls),
          verbose_(solverControls.lookupOrDefault<Switch>("verbose", true))
    {
        init_base();
    }


    //- Construct from matrix components and solver controls
    lduLduBase(const word &fieldName, const MatrixType &matrix,
               const dictionary &solverControls)
        : HostMatrix<MatrixType>(fieldName, matrix, solverControls),
          SolverFactory(solverControls, matrix.mesh().thisDb(), fieldName),
          IOSortingIdxHandler(
              matrix.mesh().thisDb(), this->nElems(),
              solverControls.lookupOrDefault<Switch>("sort", true)),
          IOGKOMatrixHandler(matrix.mesh().thisDb(), solverControls, fieldName),
          IOPreconditioner(matrix.mesh().thisDb(), solverControls, fieldName),
          solverControls_(solverControls),
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
                SIMPLE_LOG(verbose_, "host matrix needs to be resorted")
                SIMPLE_TIME(
                    verbose_, sort_host_mtx,
                    this->sort_host_matrix_data(this->get_sorting_idxs());)
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


        std::shared_ptr<mtx> gkomatrix = get_gkomatrix();

        // Generate solver
        // auto solver_gen = this->create_solver(
        //     device_exec, criterion_vec);

        // auto solver = solver_gen->generate(gko::share(gkomatrix));

        // for (int i = 0; i < 3; i++) {
        //     SIMPLE_TIME(verbose_, solve,
        //                 solver->apply(gko::lend(b[i]), gko::lend(x[i]));)
        // }

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

        auto source_view = val_array::view(ref_exec(), this->nCells(),
                                           const_cast<scalar *>(&source[0]));

        std::vector<val_array> source_views{source_view};
        SIMPLE_TIME(verbose_, create_and_push_b,
                    auto b = gko::share(
                        vec::create(device_exec, gko::dim<2>(this->nCells(), 1),
                                    source_views[0], 1));)

        init_initial_guess_vector(psi, this->matrix().mesh().thisDb(),
                                  this->nCells());
        std::vector<std::shared_ptr<vec>> x{};
        this->get_initial_guess(x);

        // Generate solver
        std::shared_ptr<mtx> gkomatrix = get_gkomatrix();

        const word timeName = this->matrix().mesh().thisDb().time().timeName();

        if (get_export())
            export_system(this->fieldName(), gko::lend(gkomatrix),
                          gko::lend(x[0]), gko::lend(b), timeName);

        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec{};

        SIMPLE_TIME(verbose_, init_preconditioner,
                    this->init_preconditioner(this->matrix().mesh().thisDb(),
                                              gkomatrix, device_exec);)

        // Instantiate an iteration logger.
        auto solver_gen =
            this->create_solver(device_exec, gkomatrix, x[0], b, verbose_,
                                this->get_export(), this->get_preconditioner());

        auto solver = solver_gen->generate(gko::share(gkomatrix));

        // Solve system
        SIMPLE_TIME(verbose_, solve,
                    solver->apply(gko::lend(b), gko::lend(x[0]));)

        this->copy_result_back(psi, this->nCells());

        solverPerf.initialResidual() = this->get_init_res_norm();
        solverPerf.finalResidual() = this->get_res_norm();
        solverPerf.nIterations() = this->get_number_of_iterations();
        this->store_number_of_iterations();

        if (get_export()) {
            export_vec(this->fieldName() + "_residuals",
                       gko::lend(this->get_res_norms()), timeName);
        }

        return solverPerf;
    };
};
}  // namespace Foam

#endif
