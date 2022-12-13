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

#include <unistd.h>
#include <fstream>
#include <iostream>

const char *git_version(void);

const char *git_revision(void);

const char *ginkgo_git_revision(void);

const char *git_branch(void);

namespace Foam {

class OGL_Info {
private:
    const objectRegistry &db_;

    const label verbose_;

    const bool debug_;

    const bool force_host_buffer_;

    const label ranks_per_gpu_;

    const word matrix_format_;

public:
    OGL_Info(const word &fieldName, const dictionary &solverControls,
             const objectRegistry &db)
        : db_(db),
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          debug_(solverControls.lookupOrDefault<Switch>("debug", false)),
          force_host_buffer_(
              solverControls.lookupOrDefault("forceHostBuffer", false)),
          ranks_per_gpu_(
              solverControls.lookupOrDefault<label>("ranksPerGPU", 1)),
          matrix_format_(
              solverControls.lookupOrDefault<word>("matrixFormat", "Coo"))
    {
        if (!db_.foundObject<regIOobject>(fieldName + "_rhs")) {
            print_info();
        }
    }

    void print_info() const
    {
        auto version_info = gko::version_info::get();

#ifndef NDEBUG
        auto build_type = std::string("Debug");
#else
        auto build_type = std::string("Release");
#endif

        // auto exec_info =
        // this->get_exec_handler().get_device_exec()->get_exec_info();
        word msg =
            std::string("Initialising OGL\n\tOGL commit: ") +
            std::string(git_version()) + std::string(" ") +
            std::string(git_revision()) + std::string("\n\tBranch: ") +
            std::string(git_branch()) + std::string(" ") +
            std::string("\n\tBuild type: ") + build_type +
            std::string("\n\tGinkgo version: ") +
            std::to_string(version_info.core_version.major) + std::string(".") +
            std::to_string(version_info.core_version.minor) + std::string(".") +
            std::to_string(version_info.core_version.patch) +
            std::string(" (") + std::string(" ") +
            std::string(version_info.core_version.tag) +
            std::string(")\n\tGinkgo commit: ") +
            std::string(ginkgo_git_revision()) +
            std::string("\n\tMPI is GPU aware: ") +
            std::to_string(gko::experimental::mpi::is_gpu_aware()) +
            std::string("\n\tForces host buffer based communication: ") +
            std::to_string(force_host_buffer_) +
            std::string("\n\tCPU ranks per GPU: ") +
            std::to_string(ranks_per_gpu_) +
            std::string("\n\tMatrix format: ") + matrix_format_ +
            std::string("\n\tEnd OGL_INFO");
        MLOG_0(verbose_, msg)

        // if in debug mode write pid
        if (debug_) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            word fn = "/tmp/mpi_debug_" + std::to_string(rank) + ".pid";
            word msg = "writing pid to" + fn;
            LOG_0(verbose_, msg)
            std::ofstream os(fn);
            os << getpid() << std::endl;
            os.close();
            sleep(20);
        }
    }
};

template <class MatrixType, class SolverFactory>
class lduLduBase : public OGL_Info,
                   public SolverFactory,
                   public HostMatrixWrapper<MatrixType>,
                   public Preconditioner {
private:
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;
    using GlobalIndexType = gko::int64;
    using part_type = gko::experimental::distributed::Partition<label, label>;
    using mtx = gko::matrix::Csr<scalar>;
    using coo_mtx = gko::matrix::Coo<scalar>;
    using csr_mtx = gko::matrix::Csr<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    using idx_array = gko::array<label>;
    using val_array = gko::array<scalar>;

    const objectRegistry &db_;

    const dictionary &solver_controls_;

    const label verbose_;

public:
    // constructor for segregated solver
    lduLduBase(const word &fieldName, const lduMatrix &matrix,
               const FieldField<Field, scalar> &interfaceBouCoeffs,
               const FieldField<Field, scalar> &interfaceIntCoeffs,
               const lduInterfaceFieldPtrsList &interfaces,
               const dictionary &solverControls)
        : OGL_Info(fieldName, solverControls, matrix.mesh().thisDb()),
          SolverFactory{solverControls, matrix.mesh().thisDb(), fieldName},
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
          solver_controls_(solverControls),
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0))
    {}


    //- Construct from matrix components and solver controls
    lduLduBase(const word &fieldName, const MatrixType &matrix,
               const dictionary &solverControls)
        : OGL_Info(fieldName, solverControls, matrix.mesh().thisDb()),
          SolverFactory{solverControls, matrix.mesh().thisDb(), fieldName},
          HostMatrixWrapper<MatrixType>(matrix.mesh().thisDb(), matrix,
                                        solverControls, fieldName),
          Preconditioner(fieldName, matrix.mesh().thisDb(), solverControls,
                         this->get_verbose()),
          db_(matrix.mesh().thisDb()),
          solver_controls_(solverControls),
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0))
    {}


    // the solve_impl_ version called from the LduMatrix, ie for
    // coupled matrices
    // TODO implement coupled solver wrapper
    template <class Type>
    SolverPerformance<Type> solve_impl_(Field<Type> &psi) const
    {}


    solverPerformance solve_multi_gpu_impl(scalarField &psi,
                                           const scalarField &source,
                                           solverPerformance &solverPerf) const
    {
        auto ref_exec = this->get_exec_handler().get_ref_exec();

        PersistentPartition partition{
            db_,
            this->get_exec_handler(),
            verbose_,
            solver_controls_.lookupOrDefault<label>("ranksPerGPU", 1),
            this->get_global_row_index().offset(Pstream::myProcNo()),  // offset
            psi.size()  // number of elements
        };

        // TODO this is now the local matrix part for dist_A
        MatrixWrapper dist_A{db_,
                             this->get_exec_handler(),  // get exec
                             this->get_local_col_idxs(),
                             this->get_local_row_idxs(),
                             this->get_local_coeffs(),
                             this->get_non_local_col_idxs(),
                             this->get_non_local_row_idxs(),
                             this->get_non_local_coeffs(),
                             partition,
                             solver_controls_,
                             this->fieldName(),
                             this->get_verbose()};

        PersistentVector<scalar> dist_b{
            &source[0],
            this->fieldName() + "_rhs",
            db_,
            this->get_exec_handler(),
            partition,
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateRHS", true),
            false  // whether data for init is on device
        };

        PersistentVector<scalar> dist_x{
            &psi[0],
            this->fieldName() + "_solution",
            db_,
            this->get_exec_handler(),
            partition,
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateInitGuess", false),
            false  // whether data for init is on device/
        };
        auto dist_x_v = dist_x.get_vector();
        auto dist_b_v = dist_b.get_vector();
        auto dist_A_v = dist_A.get();

        scalar scaling =
            solver_controls_.lookupOrDefault<scalar>("scaling", 1.0);
        if (scaling != 1) {
            auto dense_scaling =
                gko::share(gko::initialize<gko::matrix::Dense<scalar>>(
                    {scaling}, ref_exec));

            SIMPLE_TIME(verbose_, scale_RHS,
                        dist_b.get_vector()->scale(dense_scaling.get());)
        }

        SIMPLE_TIME(verbose_, init_precond,
                    auto precond = this->init_preconditioner(
                        dist_A_v, this->get_exec_handler().get_device_exec());)

        LOG_1(verbose_, "create solver")
        auto solver_gen = this->create_dist_solver(
            this->get_exec_handler().get_device_exec(), dist_A_v, dist_x_v,
            dist_b_v, verbose_, dist_A.get_export(), precond);

        SIMPLE_TIME(verbose_, generate_solver,
                    auto solver = solver_gen->generate(dist_A_v);)
        LOG_1(verbose_, "solve solver done")
        SIMPLE_TIME(verbose_, solve,
                    solver->apply(gko::lend(dist_b_v), gko::lend(dist_x_v));)

        SIMPLE_TIME(verbose_, copy_x_back, dist_x.copy_back();)
        auto bandwith_copy_back = sizeof(scalar) *
                                  partition.get_local_device_size() /
                                  delta_t_copy_x_back / 1000.0;

        solverPerf.initialResidual() = this->get_init_res_norm();
        solverPerf.finalResidual() = this->get_res_norm();
        solverPerf.nIterations() = this->get_number_of_iterations();
        this->store_number_of_iterations();
        auto time_for_res_norm_eval = this->get_res_norm_time();
        auto time_per_iter = delta_t_solve / this->get_number_of_iterations();
        scalar prev_rel_res_cost = time_per_iter / time_for_res_norm_eval;
        this->get_exec_handler().get_gko_mpi_host_comm()->broadcast(
            this->get_exec_handler().get_ref_exec(), &prev_rel_res_cost, 1, 0);
        this->set_prev_rel_res_cost(prev_rel_res_cost);
        auto time_per_iter_and_dof =
            time_per_iter * 1000.0 / partition.get_total_size();
        word msg =
            "\nStatistics:\n\tTime per iteration: " +
            std::to_string(time_per_iter) +
            std::string(" [mu s]\n\tTime per residual norm calculation: ") +
            std::to_string(time_for_res_norm_eval) +
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
            Info << "Only parallel runs are supported " << endl;
        }

        return solverPerf;
    };
};
}  // namespace Foam

#endif
