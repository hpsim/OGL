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
#include "../Preconditioner/Preconditioner.H"

#include "../common/common.H"

#include <ginkgo/ginkgo.hpp>
#include <map>

namespace Foam {

template <class MatrixType, class SolverFactory>
class lduLduBase : public SolverFactory,
                   public HostMatrixWrapper<MatrixType>,
                   public Preconditioner {
private:
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;
    using dist_vec = gko::distributed::Vector<scalar, label>;
    using dist_mtx = gko::distributed::Matrix<scalar, label>;
    using GlobalIndexType = gko::distributed::global_index_type;
    using part_type = gko::distributed::Partition<label>;
    using mtx = gko::matrix::Csr<scalar>;
    using vec = gko::matrix::Dense<scalar>;

    const objectRegistry &db_;

    const gkoGlobalIndex globalIndex_;

    const CsrMatrixWrapper csr_matrix_wrapper_;

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
          Preconditioner(fieldName, matrix.mesh().thisDb(), solverControls,
                         this->get_verbose()),
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
          Preconditioner(fieldName, matrix.mesh().thisDb(), solverControls,
                         this->get_verbose()),
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
            &source[0],
            this->fieldName() + "_rhs",
            db_,
            this->get_exec_handler(),
            this->get_global_cell_index(),
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateRHS", true),
            !solver_controls_.lookupOrDefault<Switch>("updateRHS", true),
            false};

        PersistentArray<scalar> x{
            &psi[0],
            this->fieldName() + "_solution",
            db_,
            this->get_exec_handler(),
            this->get_global_cell_index(),
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateInitGuess", false),
            !solver_controls_.lookupOrDefault<Switch>("updateInitGuess", false),
            false};

        // Solve system
        if (Pstream::parRun()) {
            SIMPLE_TIME(verbose_, gather_initGuess,
                        auto dense_x = x.get_dense_vec();)
            SIMPLE_TIME(verbose_, gather_RHS, auto dense_b = b.get_dense_vec();)

            auto ref_exec = this->get_exec_handler().get_ref_exec();


            scalar scaling =
                solver_controls_.lookupOrDefault<scalar>("scaling", 1.0);
            if (scaling != 1) {
                auto dense_scaling =
                    gko::share(gko::initialize<gko::matrix::Dense<scalar>>(
                        {scaling}, ref_exec));

                SIMPLE_TIME(verbose_, scale_RHS,
                            dense_b->scale(dense_scaling.get());)
            }

            SIMPLE_TIME(
                verbose_, gather_global_matrix,
                auto gkomatrix = csr_matrix_wrapper_.get_local_gkomatrix();)

            // build partition: uniform number of rows per rank
            auto comm = this->get_exec_handler().get_gko_mpi_comm_wrapper();
            auto rows_start =
                this->get_global_cell_index().offset(Pstream::myProcNo());
            auto rows_end = rows_start + psi.size();
            auto partition = gko::share(part_type::build_from_local_range(
                ref_exec, rows_start, rows_end, comm));


            gko::matrix_data<scalar, label> A_data_32;
            gko::matrix_data<scalar, GlobalIndexType> b_data;
            gko::matrix_data<scalar, GlobalIndexType> x_data;
            A_data_32.size = {psi.size(), psi.size()};
            b_data.size = {psi.size(), 1};
            x_data.size = {psi.size(), 1};

            auto matrix = gkomatrix.get_persistent_object();
            matrix->write(A_data_32);
            dense_b->write(b_data);
            dense_x->write(x_data);

            gko::matrix_data<scalar, GlobalIndexType> A_data;
            A_data.size = {psi.size(), psi.size()};
            for (auto &elem : A_data_32.nonzeros) {
                A_data.nonzeros.emplace_back(elem.row, elem.column, elem.value);
            }

            auto A_host =
                gko::share(dist_mtx::create(ref_exec, A_data.size, comm));
            auto b_host = dist_vec::create(ref_exec, comm);
            auto x_host = gko::share(dist_vec::create(ref_exec, comm));
            A_host->read_distributed(A_data, partition);
            b_host->read_distributed(b_data, partition);
            x_host->read_distributed(x_data, partition);

            // copy to device
            auto A =
                gko::share(dist_mtx::create(device_exec, A_data.size, comm));
            auto x = gko::share(dist_vec::create(device_exec, comm));
            auto b = gko::share(dist_vec::create(device_exec, comm));
            A->copy_from(A_host.get());
            b->copy_from(b_host.get());
            x->copy_from(x_host.get());

            LOG_1(verbose_, "create solver")
            // auto solver_gen = this->create_solver(
            //     this->get_exec_handler().get_device_exec(),
            //     gkomatrix.get_persistent_object(), x->get_local(),
            //     b->get_local(), verbose_, csr_matrix_wrapper_.get_export(),
            //     {});
            auto solver_gen = gko::share(
                gko::solver::Cg<scalar>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(100u).on(
                            device_exec))
                    .on(device_exec));

            SIMPLE_TIME(verbose_, generate_solver,
                        auto solver = solver_gen->generate(A);)
            LOG_1(verbose_, "solve solver done")
            SIMPLE_TIME(verbose_, solve,
                        solver->apply(gko::lend(b), gko::lend(x));)
            x_host->copy_from(x.get());

            // SIMPLE_TIME(verbose_, retrieve_result_from_device,
            //             dense_x->copy_back_host(&psi[0], x_host);)


            //     if (Pstream::master()) {
            //         LOG_1(verbose_, "create solver factory")


            //         SIMPLE_TIME(verbose_, init_precond,
            //                     auto precond = this->init_preconditioner(
            //                         gkomatrix.get_persistent_object(),
            //                         this->get_exec_handler().get_device_exec());)

            //         auto solver_gen = this->create_solver(
            //             this->get_exec_handler().get_device_exec(),
            //             gkomatrix.get_persistent_object(), dense_x,
            //             dense_b, verbose_,
            //             csr_matrix_wrapper_.get_export(), precond);
            //         LOG_1(verbose_, "create solver")
            //         SIMPLE_TIME(verbose_, generate_solver,
            //                     auto solver = solver_gen->generate(
            //                         gko::share(gkomatrix.get_persistent_object()));)
            //         LOG_1(verbose_, "solve solver done")
            //         SIMPLE_TIME(
            //             verbose_, solve,
            //             solver->apply(gko::lend(dense_b),
            //             gko::lend(dense_x));)
            //         LOG_1(verbose_, "copy back")
            //         solverPerf.initialResidual() =
            //         this->get_init_res_norm(); solverPerf.finalResidual()
            //         = this->get_res_norm(); solverPerf.nIterations() =
            //         this->get_number_of_iterations();
            //         this->store_number_of_iterations();
            //     }
            //     SIMPLE_TIME(verbose_, retrieve_result_from_device,
            //                 x.copy_back_parallel(this->get_global_cell_index(),
            //                                      &psi[0], dense_x);)
        } else {
            LOG_1(verbose_, "get_gkomatrix")
            auto gkomatrix = csr_matrix_wrapper_.get_local_gkomatrix();
            auto dense_x = x.get_dense_vec();
            auto dense_b = b.get_dense_vec();
            SIMPLE_TIME(verbose_, init_precond,
                        auto precond = this->init_preconditioner(
                            gkomatrix.get_persistent_object(),
                            this->get_exec_handler().get_device_exec());)
            auto solver_gen = this->create_solver(
                this->get_exec_handler().get_device_exec(),
                gkomatrix.get_persistent_object(), dense_x, dense_b, verbose_,
                csr_matrix_wrapper_.get_export(), precond);
            SIMPLE_TIME(verbose_, generate_solver,
                        auto solver = solver_gen->generate(
                            gko::share(gkomatrix.get_persistent_object()));)
            SIMPLE_TIME(verbose_, solve,
                        solver->apply(gko::lend(dense_b), gko::lend(dense_x));)
            SIMPLE_TIME(verbose_, retrieve_result_from_device,
                        x.copy_back(&psi[0], dense_x);)
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
