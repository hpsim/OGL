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


const char *git_version(void);

const char *git_revision(void);

const char *git_branch(void);

namespace Foam {

template <class MatrixType, class SolverFactory>
class lduLduBase : public SolverFactory,
                   public HostMatrixWrapper<MatrixType>,
                   public Preconditioner {
private:
    using dist_vec = gko::distributed::Vector<scalar>;
    using dist_mtx = gko::distributed::Matrix<scalar, label, label>;
    using GlobalIndexType = gko::int64;
    using part_type = gko::distributed::Partition<label, label>;
    using mtx = gko::matrix::Csr<scalar>;
    using coo_mtx = gko::matrix::Coo<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    using idx_array = gko::array<label>;
    using val_array = gko::array<scalar>;

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
        // auto exec_info =
        // this->get_exec_handler().get_device_exec()->get_exec_info();
        word msg =
            std::string("Initialising OGL\n\tVersion: ") +
            std::string(git_version()) + std::string(" ") +
            std::string(git_revision()) + std::string("\n\tBranch: ") +
            std::string(git_branch()) + std::string(" ") +
            std::string("\n\tGinkgo version: ") +
            std::to_string(version_info.core_version.major) + std::string(".") +
            std::to_string(version_info.core_version.minor) + std::string(".") +
            std::to_string(version_info.core_version.patch) + std::string("(") +
            std::string(version_info.core_version.tag) +
            std::string(")\n\tMPI is GPU aware: ") +
            std::to_string(gko::mpi::is_gpu_aware()) +
            std::string("\n\tForces host buffer based communication: ") +
            std::to_string(
                this->get_exec_handler().get_gko_force_host_buffer());
        MLOG_0(verbose_, msg)
    };


    // the solve_impl_ version called from the LduMatrix, ie for
    // coupled matrices
    // TODO implement coupled solver wrapper
    template <class Type>
    SolverPerformance<Type> solve_impl_(Field<Type> &psi) const
    {}


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
        auto bandwith_copy_back = sizeof(scalar) * psi.size() /
                                  delta_t_retrieve_result_from_device / 1000.0;

        solverPerf.initialResidual() = this->get_init_res_norm();
        solverPerf.finalResidual() = this->get_res_norm();
        solverPerf.nIterations() = this->get_number_of_iterations();
        auto time_per_iter =
            delta_t_solve * 1.0 / this->get_number_of_iterations();
        auto time_per_iter_and_dof = time_per_iter * 1000.0 / psi.size();
        word msg =
            "Time per iteration  " + std::to_string(time_per_iter) +
            std::string(" [mu s] Time per iteration and DOF ") +
            std::to_string(time_per_iter_and_dof) + std::string(" [ns]") +
            std::string(" retrieve results bandwith ") +
            std::to_string(bandwith_copy_back) + std::string(" [GByte/s]");
        MLOG_0(verbose_, msg)

        this->store_number_of_iterations();
        return solverPerf;
    }

    template <class LocMatrixType>
    std::shared_ptr<gko::distributed::Matrix<scalar, label, label>>
    init_distributed_matrix(const LocMatrixType &gkomatrix,
                            std::shared_ptr<const gko::mpi::communicator> comm,
                            std::shared_ptr<const part_type> partition,
                            const label local_size) const
    {
        auto device_exec = this->get_exec_handler().get_device_exec();
        auto csr_matrix = gkomatrix.get_persistent_object();
        auto num_elems = csr_matrix->get_num_stored_elements();

        label size = partition->get_size();
        auto coo = gko::share(
            coo_mtx::create(device_exec, gko::dim<2>(local_size, local_size)));


        auto csr_matrix_clone =
            gko::make_temporary_clone(device_exec, csr_matrix.get());
        csr_matrix_clone->convert_to(coo.get());

        gko::device_matrix_data<scalar, label> A_data{
            device_exec, gko::dim<2>{size, size},
            idx_array::view(device_exec, num_elems, coo->get_row_idxs()),
            idx_array::view(device_exec, num_elems, coo->get_col_idxs()),
            val_array::view(device_exec, num_elems, coo->get_values())};

        auto dist_A = gko::share(dist_mtx::create(device_exec, *comm.get()));
        dist_A->read_distributed(A_data, partition.get());
        return dist_A;
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
        auto device_partition = gko::share(
            gko::distributed::build_partition_from_local_range<label, label>(
                this->get_exec_handler().get_device_exec(), rows_start,
                rows_end, *comm.get()));


        PersistentVector<scalar> dist_b{
            &source[0],
            this->fieldName() + "_rhs",
            db_,
            this->get_exec_handler(),
            device_partition,
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateRHS", true),
            false  // whether data for init is on device
        };

        PersistentVector<scalar> dist_x{
            &psi[0],
            this->fieldName() + "_solution",
            db_,
            this->get_exec_handler(),
            device_partition,
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateInitGuess", false),
            false  // whether data for init is on device/
        };

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

        SIMPLE_TIME(verbose_, generate_distributed_matrix,
                    auto dist_A = init_distributed_matrix(
                        gkomatrix, comm, device_partition, psi.size());)

        SIMPLE_TIME(verbose_, init_precond,
                    auto precond = this->init_preconditioner(
                        dist_A, this->get_exec_handler().get_device_exec());)

        // check if repartitioning is requested
        label ranks_per_gpu =
            solver_controls_.lookupOrDefault<label>("RanksPerGPU", 1);
        std::shared_ptr<gko::distributed::Partition<label, label>>
            target_partition{};

        auto dist_x_v = dist_x.get_vector();
        auto dist_b_v = dist_b.get_vector();


        if (ranks_per_gpu != 1) {
            // create new target partition
            target_partition = gko::share(
                gko::distributed::build_partition_from_local_range<label,
                                                                   label>(
                    this->get_exec_handler().get_device_exec(), rows_start,
                    rows_end, *comm.get(), ranks_per_gpu));
            auto repartitioner = gko::distributed::repartitioner<label>::create(
                comm, device_partition, target_partition);
            auto to_comm = repartitioner->get_to_communicator();

            auto to_x_vec = gko::share(dist_vec::create(this->ref, to_comm));
            auto to_b_vec = gko::share(dist_vec::create(this->ref, to_comm));
            auto to_mat = gko::share(
                dist_mtx<scalar, label, label>::create(this->ref, to_comm));
            // gather data
            repartitioner->gather(dist_A.get(), to_mat.get());
            repartitioner->gather(dist_b_v.get(), to_b_vec.get());
            repartitioner->gather(dist_x_v.get(), to_x_vec.get());

            dist_x_v = to_x_vec;
            dist_b_v = to_b_vec;
            dist_A = to_mat;
        } else {
            target_partition = device_partition;
        }


        LOG_1(verbose_, "create solver")
        auto solver_gen = this->create_dist_solver(
            this->get_exec_handler().get_device_exec(), dist_A, dist_x_v,
            dist_b_v, verbose_, csr_matrix_wrapper_.get_export(), precond);

        SIMPLE_TIME(verbose_, generate_solver,
                    auto solver = solver_gen->generate(dist_A);)
        LOG_1(verbose_, "solve solver done")
        SIMPLE_TIME(verbose_, solve,
                    solver->apply(gko::lend(dist_b_v), gko::lend(dist_x_v));)

        // check if scattering is nedded

        SIMPLE_TIME(verbose_, copy_x_back, dist_x.copy_back();)
        auto bandwith_copy_back =
            sizeof(scalar) * psi.size() / delta_t_copy_x_back / 1000.0;

        solverPerf.initialResidual() = this->get_init_res_norm();
        solverPerf.finalResidual() = this->get_res_norm();
        solverPerf.nIterations() = this->get_number_of_iterations();
        this->store_number_of_iterations();
        auto time_per_iter = delta_t_solve / this->get_number_of_iterations();
        auto time_per_iter_and_dof =
            time_per_iter * 1000.0 / target_partition->get_size();
        word msg =
            "\nStatistics:\n\tTime per iteration: " +
            std::to_string(time_per_iter) +
            std::string(" [mu s]\n\tTime per iteration and DOF: ") +
            std::to_string(time_per_iter_and_dof) + std::string(" [ns]") +
            std::string("\n\tRetrieve results bandwith ") +
            std::to_string(bandwith_copy_back) + std::string(" [GByte/s]");
        MLOG_0(verbose_, msg)

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
