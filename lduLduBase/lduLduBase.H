// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "DevicePersistent/HostMatrixAdapter/HostMatrixAdapter.H"
#include "DevicePersistent/DistributedMatrixAdapter/DistributedMatrixAdapter.H"
#include "DevicePersistent/Vector/Vector.H"
#include "Preconditioner/Preconditioner.H"
#include "common/common.H"

#include <ginkgo/ginkgo.hpp>
#include <map>

#include <string.h>
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

    // const label ranks_per_gpu_;

    const word matrix_format_;

public:
    OGL_Info(const word &fieldName, const dictionary &solverControls,
             const objectRegistry &db)
        : db_(db),
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          debug_(solverControls.lookupOrDefault<Switch>("debug", false)),
          force_host_buffer_(
              solverControls.lookupOrDefault("forceHostBuffer", false)),
          // ranks_per_gpu_(
          //     solverControls.lookupOrDefault<label>("ranksPerGPU", 1)),
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
            // std::string("\n\tCPU ranks per GPU: ") +
            // std::to_string(ranks_per_gpu_) +
            std::string("\n\tMatrix format: ") + matrix_format_ +
            std::string("\n\tEnd OGL_INFO");
        MLOG_0(verbose_, msg)

        // if in debug mode write pid
        if (debug_) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            word fn = "/tmp/mpi_debug_" + std::to_string(rank) + ".pid";
            word msg =
                "writing pid to" + fn + " waiting 20s for debugger to attach";
            LOG_0(verbose_, msg)
            std::ofstream os(fn);
            os << getpid() << std::endl;
            os.close();
            sleep(20);
        }
    }
};

/**
 * Base class for all solver
 *
 */
template <class MatrixType, class SolverFactory>
class lduLduBase : public OGL_Info,
                   public MatrixType::solver,
                   public SolverFactory,
                   public Preconditioner {
private:
    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    using mtx = gko::matrix::Csr<scalar>;
    using coo_mtx = gko::matrix::Coo<scalar>;
    using csr_mtx = gko::matrix::Csr<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    using idx_array = gko::array<label>;
    using val_array = gko::array<scalar>;

    const objectRegistry &db_;

    const MatrixType &matrix_;

    const dictionary &solver_controls_;

    const word & field_name_;

    const label verbose_;

    const ExecutorHandler exec_handler_;

    // A thin wrapper around OpenFOAMs LDU matrix class
    PersistentHostMatrixAdapter host_A_;

public:
    // constructor for segregated solver
    lduLduBase(const word &fieldName, const lduMatrix &matrix,
               const FieldField<Field, scalar> &interfaceBouCoeffs,
               const FieldField<Field, scalar> &interfaceIntCoeffs,
               const lduInterfaceFieldPtrsList &interfaces,
               const dictionary &solverControls)
        : OGL_Info(fieldName, solverControls, matrix.mesh().thisDb()),
          MatrixType::solver(fieldName, matrix, interfaceBouCoeffs,
                         interfaceIntCoeffs, interfaces, solverControls),
          SolverFactory{solverControls, matrix.mesh().thisDb(), fieldName},
          Preconditioner(fieldName, matrix.mesh().thisDb(), solverControls,
                solverControls.lookupOrDefault<label>("verbose", 0)     
                  ),
          db_(matrix.mesh().thisDb()),
          matrix_(matrix),
          solver_controls_(solverControls),
          field_name_(fieldName),
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          exec_handler_{
              db_, solver_controls_, fieldName
          },
          host_A_{
              db_,
              exec_handler_,
              solver_controls_,
              fieldName,
              verbose_,
              matrix_,
              interfaceBouCoeffs,
              interfaceIntCoeffs,
              interfaces
          }
    {}


    /**
     * Constructor from matrix components and solver controls
     *
     */
    lduLduBase(const word &fieldName, const MatrixType &matrix,
               const dictionary &solverControls)
        : OGL_Info(fieldName, solverControls, matrix.mesh().thisDb()),
          SolverFactory{solverControls, matrix.mesh().thisDb(), fieldName},
          // HostMatrixWrapper<MatrixType>(matrix.mesh().thisDb(), matrix,
          //                               solverControls, fieldName),
          Preconditioner(fieldName, matrix.mesh().thisDb(), solverControls,
                         verbose_),
          db_(matrix.mesh().thisDb()),
          matrix_(matrix),
          solver_controls_(solverControls),
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0))
    {}


    // the solve_impl_ version called from the LduMatrix, ie for
    // coupled matrices
    // TODO implement coupled solver wrapper
    template <class Type>
    SolverPerformance<Type> solve_impl_(Field<Type> &psi) const
    {}


    solverPerformance solve_multi_gpu_impl(
            ExecutorHandler &exec_handler,
            scalarField &psi,
                                           const scalarField &source,
                                           solverPerformance &solverPerf) const
    {
        auto ref_exec = exec_handler.get_ref_exec();

        // create a distributed matrix from the host matrix
        // here also repartitioning is performed if required
        PersistentDistributedMatrix dist_A{db_,
                             exec_handler,
                             field_name_,
                             host_A_,
                             solver_controls_,
                             verbose_};

        PersistentVector<scalar> dist_b{
            source.begin(),
            source.size(),
            field_name_ + "_rhs",
            db_,
            exec_handler,
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateRHS", true),
            false  // whether data for init is on device
        };

        PersistentVector<scalar> dist_x{
            psi.begin(),
            psi.size(),
            field_name_ + "_solution",
            db_,
            exec_handler,
            verbose_,
            solver_controls_.lookupOrDefault<Switch>("updateInitGuess", false),
            false  // whether data for init is on device
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

            TIME_WITH_FIELDNAME(
                verbose_, scale_RHS, field_name_,
                dist_b.get_vector()->scale(dense_scaling.get());)
        }

        TIME_WITH_FIELDNAME(
            verbose_, init_precond, field_name_,
            auto precond = this->init_preconditioner(
                dist_A_v, exec_handler.get_device_exec());)

        bool write(solver_controls_.lookupOrDefault<Switch>("export", false));
        if (write && db_.time().writeTime()) {
            LOG_0(verbose_, "Export system")
            dist_A.get()->write(field_name_, db_);
            dist_b.write();
        }


        LOG_1(verbose_, "create solver")
        auto solver_gen = this->create_dist_solver(
            exec_handler.get_device_exec(), dist_A_v, dist_x_v,
            dist_b_v, verbose_, write, precond);

        TIME_WITH_FIELDNAME(verbose_, generate_solver, field_name_,
                            auto solver = solver_gen->generate(dist_A_v);)
        LOG_1(verbose_, "generate solver done")
        TIME_WITH_FIELDNAME(verbose_, solve, field_name_,
                            solver->apply(dist_b_v, dist_x_v);)

        auto partition = dist_A.get()->get_partition();
        TIME_WITH_FIELDNAME(verbose_, copy_x_back, field_name_,
                            dist_x.copy_back(partition);)
        auto bandwidth_copy_back = sizeof(scalar) * psi.size() /
                                   (delta_t_copy_x_back + SMALL) / 1000.0;

        solverPerf.initialResidual() = this->get_init_res_norm();
        solverPerf.finalResidual() = this->get_res_norm();
        solverPerf.nIterations() = this->get_number_of_iterations();
        this->store_number_of_iterations();
        auto time_for_res_norm_eval = this->get_res_norm_time();
        auto time_per_iter =
            delta_t_solve / max(this->get_number_of_iterations(), 1);
        scalar prev_rel_res_cost = time_per_iter / time_for_res_norm_eval;
        exec_handler.get_gko_mpi_host_comm()->broadcast(
            exec_handler.get_ref_exec(), &prev_rel_res_cost, 1, 0);
        this->set_prev_rel_res_cost(prev_rel_res_cost);
        auto time_per_iter_and_dof =
            time_per_iter * 1000.0 / partition->get_size();
        word msg =
            "\nStatistics:\n\tTime per iteration: " +
            std::to_string(time_per_iter) +
            std::string(" [mu s]\n\tTime per residual norm calculation: ") +
            std::to_string(time_for_res_norm_eval) +
            std::string(" [mu s]\n\tTime per iteration and DOF: ") +
            std::to_string(time_per_iter_and_dof) + std::string(" [ns]") +
            std::string("\n\tRetrieve results bandwidth ") +
            std::to_string(bandwidth_copy_back) + std::string(" [GByte/s]");
        MLOG_0(verbose_, msg)

        return solverPerf;
    }

    solverPerformance solve_impl_(word typeName, scalarField &psi,
                                  const scalarField &source,
                                  const direction cmpt = 0) const
    {
        if (!Pstream::parRun()) {
            FatalErrorInFunction << "Only parallel runs are supported for OGL"
                                 << exit(FatalError);
        }
        auto exec_handler = host_A_.get()->get_exec_handler();
        // --- Setup class containing solver performance data
        solverPerformance solverPerf(
            lduMatrix::preconditioner::getName(solver_controls_) +
                exec_handler.get_exec_name() + typeName,
            field_name_);

        // Solve system
        TIME_WITH_FIELDNAME(
            verbose_, solve_multi_gpu, field_name_,
            auto res = solve_multi_gpu_impl(exec_handler, psi, source, solverPerf);)
        return res;

        return solverPerf;
    };
};
}  // namespace Foam
