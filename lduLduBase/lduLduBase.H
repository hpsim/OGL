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
    using dist_vec = gko::distributed::Vector<scalar>;
    using dist_mtx = gko::distributed::Matrix<scalar, label>;
    using GlobalIndexType = gko::int64;
    using part_type = gko::distributed::Partition<label>;
    using mtx = gko::matrix::Csr<scalar>;
    using vec = gko::matrix::Dense<scalar>;

    const objectRegistry &db_;

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
        if (!db_.foundObject<regIOobject>(fieldName + "_rhs")) {
            print_info();
        }
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
    {}

    void print_info() const
    {
        auto version_info = gko::version_info::get();

        word msg =
            std::string("Initialising OGL\n") +
            std::string("\t Ginkgoversion") +
            std::to_string(version_info.core_version.major) + std::string(".") +
            std::to_string(version_info.core_version.minor) + std::string(".") +
            std::to_string(version_info.core_version.patch) + std::string("(") +
            std::string(version_info.core_version.tag) +
            std::string(")\n\tMPI is GPU aware: ") +
            std::to_string(gko::mpi::is_gpu_aware());
        LOG_1(verbose_, msg)
    };


    // the solve_impl_ version called from the LduMatrix, ie for
    // coupled matrices
    // TODO implement coupled solver wrapper
    template <class Type>
    SolverPerformance<Type> solve_impl_(Field<Type> &psi) const {};


    solverPerformance solve_serial_impl(scalarField &psi,
                                        const scalarField &source,
                                        solverPerformance &solverPerf) const
    {
        PersistentArray<scalar> b{
            const_cast<scalar *>(&source[0]),
            this->fieldName() + "_rhs",
            db_,
            this->get_exec_handler(),
            this->get_global_cell_index(),
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateRHS", true),
            false};

        PersistentArray<scalar> x{
            &psi[0],
            this->fieldName() + "_solution",
            db_,
            this->get_exec_handler(),
            this->get_global_cell_index(),
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateInitGuess", false),
            false};

        auto ref_exec = this->get_exec_handler().get_ref_exec();
        LOG_1(verbose_, "get_gkomatrix")
        auto gkomatrix = csr_matrix_wrapper_.get_local_gkomatrix();
        auto dense_x = x.get_dense_vec();
        auto dense_b = b.get_dense_vec();

        scalar scaling =
            solver_controls_.lookupOrDefault<scalar>("scaling", 1.0);

        if (scaling != 1) {
            auto dense_scaling =
                gko::share(gko::initialize<gko::matrix::Dense<scalar>>(
                    {scaling}, ref_exec));

            // TODO FIXME fails cudaErrorIllegalAddress
            SIMPLE_TIME(verbose_, scale_RHS,
                        dense_b->scale(dense_scaling.get());)
        }

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
                    x.copy_back(dense_x);)
        solverPerf.initialResidual() = this->get_init_res_norm();
        solverPerf.finalResidual() = this->get_res_norm();
        solverPerf.nIterations() = this->get_number_of_iterations();
        this->store_number_of_iterations();
        return solverPerf;
    }

    solverPerformance solve_multi_gpu_impl(scalarField &psi,
                                           const scalarField &source,
                                           solverPerformance &solverPerf) const
    {
        // build partition: uniform number of rows per rank
        auto comm = this->get_exec_handler().get_gko_mpi_comm_wrapper();
        auto rows_start =
            this->get_global_cell_index().offset(Pstream::myProcNo());
        auto rows_end = rows_start + psi.size();

        auto ref_exec = this->get_exec_handler().get_ref_exec();
        auto partition = gko::share(
            gko::distributed::build_partition_from_local_range<label,
                                                               gko::int64>(
                ref_exec, rows_start, rows_end, *comm.get()));

        PersistentVector<scalar> dist_b{
            &source[0],
            this->fieldName() + "_rhs",
            db_,
            this->get_exec_handler(),
            partition,
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateRHS", true),
            false};

        PersistentVector<scalar> dist_x{
            &psi[0],
            this->fieldName() + "_solution",
            db_,
            this->get_exec_handler(),
            partition,
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateInitGuess", false),
            false};

        scalar scaling =
            solver_controls_.lookupOrDefault<scalar>("scaling", 1.0);
        if (scaling != 1) {
            auto dense_scaling =
                gko::share(gko::initialize<gko::matrix::Dense<scalar>>(
                    {scaling}, ref_exec));

            SIMPLE_TIME(verbose_, scale_RHS,
                        dist_b.get_vector()->scale(dense_scaling.get());)
        }

        SIMPLE_TIME(verbose_, gather_global_matrix,
                    auto gkomatrix = csr_matrix_wrapper_.get_local_gkomatrix();)


        gko::matrix_data<scalar, label> A_data_32;
        A_data_32.size = {psi.size(), psi.size()};

        auto matrix = gkomatrix.get_persistent_object();
        matrix->write(A_data_32);

        gko::matrix_data<scalar, GlobalIndexType> A_data;
        const label global_size = this->get_global_cell_index().size();
        A_data.size = {global_size, global_size};
        for (auto &elem : A_data_32.nonzeros) {
            A_data.nonzeros.emplace_back(elem.row, elem.column, elem.value);
        }

        auto device_exec = this->get_exec_handler().get_device_exec();

        auto A_host = gko::share(dist_mtx::create(ref_exec, *comm.get()));
        A_host->read_distributed(A_data, partition.get());

        /* // copy to device */
        auto dist_A = gko::share(dist_mtx::create(device_exec, *comm.get()));
        dist_A->copy_from(A_host.get());

        SIMPLE_TIME(verbose_, init_precond,
                    auto precond = this->init_preconditioner(
                        dist_A, this->get_exec_handler().get_device_exec());)


        LOG_1(verbose_, "create solver")
        auto solver_gen = this->create_dist_solver(
            this->get_exec_handler().get_device_exec(), dist_A,
            dist_x.get_vector(), dist_b.get_vector(), verbose_,
            csr_matrix_wrapper_.get_export(), precond);

        /* auto solver_gen = gko::share( */
        /*     gko::solver::Cg<scalar>::build() */
        /*         .with_criteria( */
        /*             gko::stop::Iteration::build() */
        /*                 .with_max_iters( */
        /*                     solver_controls_.lookupOrDefault("maxIter",
         * 1000)) */
        /*                 .on(device_exec)) */
        /*         .on(device_exec)); */

        SIMPLE_TIME(verbose_, generate_solver,
                    auto solver = solver_gen->generate(dist_A);)
        LOG_1(verbose_, "solve solver done")
        SIMPLE_TIME(verbose_, solve,
                    solver->apply(gko::lend(dist_b.get_vector()),
                                  gko::lend(dist_x.get_vector()));)


        SIMPLE_TIME(verbose_, copy_x_back, dist_x.copy_back();)

        solverPerf.initialResidual() = this->get_init_res_norm();
        solverPerf.finalResidual() = this->get_res_norm();
        solverPerf.nIterations() = this->get_number_of_iterations();
        this->store_number_of_iterations();

        return solverPerf;
    }

    solverPerformance solve_impl_(word typeName, scalarField &psi,
                                  const scalarField &source,
                                  const direction cmpt = 0) const
    {
        // --- Setup class containing solver performance data
        solverPerformance solverPerf(
            lduMatrix::preconditioner::getName(this->controlDict_) +
                this->get_exec_handler().get_exec_name() + typeName,
            this->fieldName());

        // Solve system
        if (Pstream::parRun()) {
            return solve_multi_gpu_impl(psi, source, solverPerf);
        } else {
            return solve_serial_impl(psi, source, solverPerf);
        }

        return solverPerf;
    };
};
}  // namespace Foam

#endif
