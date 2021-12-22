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
    lduLduBase.H

\*---------------------------------------------------------------------------*/

#ifndef OGL_lduLduBase_INCLUDED_H
#define OGL_lduLduBase_INCLUDED_H

#include "../../HostMatrix/HostMatrix.H"
#include "../CsrMatrixWrapper/CsrMatrixWrapper.H"
#include "../ExecutorHandler/ExecutorHandler.H"
#include "../IOGlobalIndex/gkoGlobalIndex.H"

#include "../common/common.H"

#include <ginkgo/ginkgo.hpp>
#include <map>

namespace Foam {

template <class MatrixType, class SolverFactory>
class lduLduBase : public SolverFactory,
                   public HostMatrixWrapper<MatrixType>  //,
//                   public IOPreconditioner
{
private:
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;
    using mtx = gko::matrix::Csr<scalar>;
    using vec = gko::matrix::Dense<scalar>;


    const objectRegistry &db_;

    const gkoGlobalIndex globalIndex_;

    const CsrMatrixWrapper csr_matrix_wrapper_;

    // TODO rename to Preconditioner factory
    // const IOPreconditioner gkoPreconditionerFactory_;

    const dictionary &solver_controls_;

    const label verbose_;

public:
    // constructor for segregated solver
    lduLduBase(const word &fieldName, const lduMatrix &matrix,
               const FieldField<Field, scalar> &interfaceBouCoeffs,
               const FieldField<Field, scalar> &interfaceIntCoeffs,
               const lduInterfaceFieldPtrsList &interfaces,
               const dictionary &solverControls)
        : SolverFactory{solverControls, matrix.mesh().thisDb(), fieldName},
          HostMatrixWrapper<MatrixType>{matrix.mesh().thisDb(),
                                        matrix,
                                        interfaceBouCoeffs,
                                        interfaceIntCoeffs,
                                        interfaces,
                                        solverControls,
                                        fieldName},
          db_(matrix.mesh().thisDb()),
          csr_matrix_wrapper_{db_,  // 1
                              this->get_exec_handler(),
                              this->get_col_idxs(),
                              this->get_row_idxs(),
                              this->get_values(),
                              solverControls,
                              fieldName,
                              this->get_global_cell_index(),
                              this->get_verbose()},
          // gkoPreconditionerFactory_{db_, solverControls, fieldName},
          solver_controls_(solverControls),
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0))
    {
        init_base();
    }


    //- Construct from matrix components and solver controls
    lduLduBase(const word &fieldName, const MatrixType &matrix,
               const dictionary &solverControls)
        : SolverFactory{solverControls, matrix.mesh().thisDb(), fieldName},
          HostMatrixWrapper<MatrixType>(matrix.mesh().thisDb(), matrix,
                                        solverControls, fieldName),
          db_(matrix.mesh().thisDb()),
          csr_matrix_wrapper_{db_,  // 1
                              this->get_exec_handler(),
                              this->get_col_idxs(),
                              this->get_row_idxs(),
                              this->get_values(),
                              solverControls,
                              fieldName,
                              this->get_global_cell_index(),
                              this->get_verbose()},
          // gkoPreconditionerFactory_{db_, solverControls, fieldName},
          solver_controls_(solverControls),
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0))
    {
        init_base();
    }

    void init_base() {}


    // the solve_impl_ version called from the LduMatrix, ie for
    // coupled matrices
    // TODO implement coupled solver wrapper
    template <class Type>
    SolverPerformance<Type> solve_impl_(Field<Type> &psi) const {};

    solverPerformance solve_impl_(word typeName, scalarField &psi,
                                  const scalarField &source,
                                  const direction cmpt = 0) const
    {
        std::shared_ptr<gko::Executor> device_exec =
            this->get_exec_handler().get_device_exec();

        // --- Setup class containing solver performance data
        solverPerformance solverPerf(
            lduMatrix::preconditioner::getName(this->controlDict_) +
                this->get_exec_handler().get_exec_name() + typeName,
            this->fieldName());


        PersistentArray<scalar> b{
            this->fieldName() + "_rhs",
            db_,
            this->get_exec_handler(),
            this->get_global_cell_index(),
            &source[0],
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateRHS", true),
            true};

        PersistentArray<scalar> x{
            this->fieldName() + "_solution",
            db_,
            this->get_exec_handler(),
            this->get_global_cell_index(),
            &psi[0],
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateInitGuess", true),
            true};


        // Generate solver
        // const word timeName =
        // this->matrix().mesh().thisDb().time().timeName();

        // // if (Pstream::master()) {
        // //     if (get_export())
        // //         export_system(this->fieldName(), gko::lend(gkomatrix),
        // //                       gko::lend(x[0]), gko::lend(b), timeName);
        // // }

        // SIMPLE_LOG(verbose_, "create precond")
        // if (Pstream::master()) {
        //     SIMPLE_TIME(
        //         verbose_, init_preconditioner,
        //         this->init_preconditioner(this->matrix().mesh().thisDb(),
        //                                   gkomatrix, device_exec);)
        // }
        // SIMPLE_LOG(verbose_, "create precond done")

        // // Instantiate an iteration logger.


        // Solve system
        if (Pstream::parRun()) {
            auto dense_x = x.get_global_dense_vec();
            auto dense_b = b.get_global_dense_vec();
            LOG_1(verbose_, "get_gkomatrix")
            auto gkomatrix = csr_matrix_wrapper_.get_global_gkomatrix();
            if (Pstream::master()) {
                LOG_1(verbose_, "create solver factory")

                auto solver_gen = this->create_solver(
                    this->get_exec_handler().get_device_exec(),
                    gkomatrix.get_persistent_object(), dense_x, dense_b,
                    verbose_, csr_matrix_wrapper_.get_export(),
                    // this->get_preconditioner()
                    std::shared_ptr<gko::LinOp>{});
                LOG_1(verbose_, "create solver")
                auto solver = solver_gen->generate(
                    gko::share(gkomatrix.get_persistent_object()));
                LOG_1(verbose_, "solve solver")
                SIMPLE_TIME(
                    verbose_, solve,
                    solver->apply(gko::lend(dense_b), gko::lend(dense_x));)
                LOG_1(verbose_, "copy back")
                solverPerf.initialResidual() = this->get_init_res_norm();
                solverPerf.finalResidual() = this->get_res_norm();
                solverPerf.nIterations() = this->get_number_of_iterations();
                this->store_number_of_iterations();
            }
            x.copy_back_parallel(this->get_global_cell_index(), &psi[0],
                                 dense_x);
        } else {
            LOG_1(verbose_, "get_gkomatrix")
            auto gkomatrix = csr_matrix_wrapper_.get_gkomatrix();

            auto dense_x = x.get_dense_vec();
            auto dense_b = b.get_dense_vec();
            auto solver_gen = this->create_solver(
                this->get_exec_handler().get_device_exec(),
                gkomatrix.get_persistent_object(), dense_x, dense_b, verbose_,
                csr_matrix_wrapper_.get_export(),
                // this->get_preconditioner()
                std::shared_ptr<gko::LinOp>{});
            auto solver = solver_gen->generate(
                gko::share(gkomatrix.get_persistent_object()));
            SIMPLE_TIME(verbose_, solve,
                        solver->apply(gko::lend(dense_b), gko::lend(dense_x));)
            x.copy_back(&psi[0], dense_x);
            solverPerf.initialResidual() = this->get_init_res_norm();
            solverPerf.finalResidual() = this->get_res_norm();
            solverPerf.nIterations() = this->get_number_of_iterations();
            this->store_number_of_iterations();
        }

        return solverPerf;
    };
};
}  // namespace Foam

#endif
