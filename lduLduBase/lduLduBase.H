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
#include "../IOExecutorHandler/IOExecutorHandler.H"
#include "../IOGKOMatrixHandler/IOGKOMatrixHandler.H"
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

    const bool verbose_;

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
                              this->get_verbose()},
          // gkoPreconditionerFactory_{db_, solverControls, fieldName},
          solver_controls_(solverControls),
          verbose_(solverControls.lookupOrDefault<Switch>("verbose", true))
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
                              this->get_verbose()},
          // gkoPreconditionerFactory_{db_, solverControls, fieldName},
          solver_controls_(solverControls),
          verbose_(solverControls.lookupOrDefault<Switch>("verbose", true))
    {
        init_base();
    }

    void init_base()
    {
        // TODO remove since this is moved to host matrix now
        // if sys_matrix is not stored updating is neccesary
        // initially
        // bool stored = get_sys_matrix_stored();
        // if (!stored) {
        //     SIMPLE_TIME(verbose_, init_host_sparsity_pattern,
        //                 this->init_host_sparsity_pattern();)


        //     SIMPLE_LOG(verbose_, "matrix not stored update host matrix")
        //     SIMPLE_TIME(verbose_, update_host_mtx,
        //                 this->update_host_matrix_data();)
        // } else {
        //     // if sys_matrix is  stored updating is only neccesary
        //     // when requested explictly
        //     if (get_update_sys_matrix()) {
        //         SIMPLE_LOG(
        //             verbose_,
        //             "matrix is stored and update of host matrix requested")
        //         SIMPLE_TIME(verbose_, exp_update_host_mtx,
        //                     this->update_host_matrix_data();)
        //     }
        // }
        // auto globalAddr = gkoGlobalIndex(this->nCells());


        // SIMPLE_TIME(
        //     verbose_, init_device_matrix,
        //     init_device_matrix(this->matrix().mesh().thisDb(), globalAddr,
        //                        this->values(), this->col_idxs(),
        //                        this->row_idxs(), this->nElems(),
        //                        this->nCells(), get_update_sys_matrix());)
    }


    // template <class OFField>
    // void init_vector_views(std::vector<val_array> &target,
    //                        OFField &source) const
    // {
    //     for (int i = 0; i < 3; i++) {
    //         target.push_back(
    //             val_array::view(ref_exec(), 3 * this->nCells(),
    //                             const_cast<scalar *>(&source[0][i])));
    //     }
    // }

    // the solve_impl_ version called from the LduMatrix, ie for
    // coupled matrices
    template <class Type>
    SolverPerformance<Type> solve_impl_(Field<Type> &psi) const {
        // auto gI = globalIndex(this->nCells());
        // Info << "global size" << gI.size() << endl;


        // std::shared_ptr<gko::Executor> device_exec =
        //     this->get_device_executor();

        // // --- Setup class containing solver performance data
        // // Implement
        // word preconditionerName(this->controlDict_.lookup("preconditioner"));

        // SolverPerformance<Type> solverPerf(
        //     this->get_device_executor_name() + preconditionerName,
        //     this->fieldName_);

        // std::vector<val_array> source_views{};

        // init_vector_views(source_views, gI, this->matrix().source());

        // std::vector<std::shared_ptr<vec>> b{};
        // for (int i = 0; i < 3; i++) {
        //     b.push_back(vec::create(ref_exec(), gko::dim<2>(this->nCells(),
        //     1),
        //                             source_views[i], 3));
        // }

        // init_initial_guess_vector(psi, this->matrix().mesh().thisDb(),
        //                           this->nCells());
        // std::vector<std::shared_ptr<vec>> x{};
        // this->get_initial_guess(x);

        // std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        //     criterion_vec{};


        // std::shared_ptr<mtx> gkomatrix = get_gkomatrix();

        // // Generate solver
        // // auto solver_gen = this->create_solver(
        // //     device_exec, criterion_vec);

        // // auto solver = solver_gen->generate(gko::share(gkomatrix));

        // // for (int i = 0; i < 3; i++) {
        // //     SIMPLE_TIME(verbose_, solve,
        // //                 solver->apply(gko::lend(b[i]), gko::lend(x[i]));)
        // // }

        // // copy back
        // copy_result_back(psi, this->nCells());

        // return solverPerf;
    };

    solverPerformance solve_impl_(word typeName, scalarField &psi,
                                  const scalarField &source,
                                  const direction cmpt = 0) const
    {
        std::shared_ptr<gko::Executor> device_exec =
            this->get_exec_handler().get_device_exec();

        // --- Setup class containing solver performance data
        solverPerformance solverPerf(
            lduMatrix::preconditioner::getName(this->controlDict_) +
                this->get_exec_handler().get_device_executor_name() + typeName,
            this->fieldName());


        PersistentArray<scalar> b{
            this->fieldName() + "rhs",
            db_,
            this->get_exec_handler(),
            this->get_global_cell_index(),
            &source[0],
            solver_controls_.lookupOrDefault<Switch>("updateRHS", true),
            true};

        PersistentArray<scalar> x{
            this->fieldName() + "solution",
            db_,
            this->get_exec_handler(),
            this->get_global_cell_index(),
            &psi[0],
            solver_controls_.lookupOrDefault<Switch>("updateInitGuess", true),
            true};


        // Generate solver
        SIMPLE_LOG(verbose_, "get_gkomatrix")
        auto gkomatrix = csr_matrix_wrapper_.get_gkomatrix();

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
            // if (Pstream::master()) {
            //     SIMPLE_LOG(verbose_, "init_initial_guess done")
            //     std::vector<std::shared_ptr<vec>> x{};
            //     this->get_initial_guess(x);
            //     SIMPLE_LOG(verbose_, "get init_initial_guess done")
            //     SIMPLE_LOG(verbose_, "create solver")
            //     auto solver_gen = this->create_solver(
            //         device_exec, gkomatrix, x[0], b, verbose_,
            //         this->get_export(), this->get_preconditioner());
            //     auto solver = solver_gen->generate(gko::share(gkomatrix));
            //     SIMPLE_TIME(verbose_, solve,
            //                 solver->apply(gko::lend(b), gko::lend(x[0]));)
            //     //        this->copy_result_back(psi, this->nCells());
            //     SIMPLE_LOG(verbose_, "create solver done")
            //     if (get_export()) {
            //         export_vec(this->fieldName() + "_residuals",
            //                    gko::lend(this->get_res_norms()), timeName);
            //     }
            //     solverPerf.initialResidual() = this->get_init_res_norm();
            //     solverPerf.finalResidual() = this->get_res_norm();
            //     solverPerf.nIterations() = this->get_number_of_iterations();
            //     this->store_number_of_iterations();
            //     this->copy_result_back(globalAddr, psi);
            // }
        } else {
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
            // this->copy_result_back(psi, this->nCells());
            //
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
