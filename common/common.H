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
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/
#ifndef OGL_COMMON_H
#define OGL_COMMON_H

#include "fvCFD.H"
#include "regIOobject.H"

#include "../IOExecutorHandler/IOExecutorHandler.H"
#include "../IOGKOMatrixHandler/IOGKOMatrixHandler.H"
#include "../IOHandler/IOPreconditioner/IOPreconditioner.H"
#include "../IOHandler/IOSortingIdxHandler/IOSortingIdxHandler.H"
#include "../common/StoppingCriterion.H"

#include <ginkgo/ginkgo.hpp>
#include <map>

#define SIMPLE_TIME(NAME, F)                                            \
    auto start_##NAME = std::chrono::steady_clock::now();               \
    F auto end_##NAME = std::chrono::steady_clock::now();               \
    std::cout << "Gingko " #NAME " : "                                  \
              << std::chrono::duration_cast<std::chrono::milliseconds>( \
                     end_##NAME - start_##NAME)                         \
                     .count()                                           \
              << " ms\n";

namespace Foam {

template <class MatrixType>
class HostMatrix : public MatrixType::solver {
private:
    const label nCells_;

    const label nNeighbours_;

    const label nElems_;

    mutable std::vector<scalar> values_;

    mutable std::vector<label> col_idxs_;

    mutable std::vector<label> row_idxs_;

public:
    HostMatrix(const word &fieldName, const MatrixType &matrix,
               const FieldField<Field, scalar> &interfaceBouCoeffs,
               const FieldField<Field, scalar> &interfaceIntCoeffs,
               const lduInterfaceFieldPtrsList &interfaces,
               const dictionary &solverControls)
        : MatrixType::solver(fieldName, matrix, interfaceBouCoeffs,
                             interfaceIntCoeffs, interfaces, solverControls),
          nCells_(matrix.diag().size()),
          nNeighbours_(matrix.lduAddr().upperAddr().size()),
          nElems_(nCells_ + 2 * nNeighbours_){};

    HostMatrix(const word &fieldName, const MatrixType &matrix,
               const dictionary &solverControls)
        : MatrixType::solver(fieldName, matrix, solverControls),
          nCells_(matrix.diag().size()),
          nNeighbours_(matrix.lduAddr().upperAddr().size()),
          nElems_(nCells_ + 2 * nNeighbours_){};

    void init_host_sparsity_pattern() const
    {
        col_idxs_.resize(0);
        row_idxs_.resize(0);

        col_idxs_.reserve(nElems());
        row_idxs_.reserve(nElems());

        // fill vectors unsorted
        for (label i = 0; i < nNeighbours(); ++i) {
            row_idxs_.push_back(this->matrix().lduAddr().lowerAddr()[i]);
            col_idxs_.push_back(this->matrix().lduAddr().upperAddr()[i]);
        }

        for (label i = 0; i < nCells(); ++i) {
            col_idxs_.push_back(i);
            row_idxs_.push_back(i);
        }

        for (label i = 0; i < nNeighbours(); ++i) {
            row_idxs_.push_back(this->matrix().lduAddr().upperAddr()[i]);
            col_idxs_.push_back(this->matrix().lduAddr().lowerAddr()[i]);
        }
    }

    void sort_host_matrix_sparsity_pattern(
        const IOField<label> *sorting_idxs) const
    {
        std::vector<label> tmp_col_idxs(nElems());
        std::vector<label> tmp_row_idxs(nElems());

        for (label i = 0; i < nElems(); i++) tmp_col_idxs[i] = col_idxs_[i];
        for (label i = 0; i < nElems(); i++) tmp_row_idxs[i] = row_idxs_[i];

        for (label i = 0; i < nElems(); i++) {
            label j = sorting_idxs->operator[](i);
            col_idxs_[i] = tmp_col_idxs[j];
            row_idxs_[i] = tmp_row_idxs[j];
        }
    };


    void update_host_matrix_data() const
    {
        // reset vectors
        values_.resize(0);

        values_.reserve(nElems());

        // fill vectors unsorted
        for (label i = 0; i < nNeighbours(); ++i) {
            values_.push_back(this->matrix().lower()[i]);
        }

        for (label i = 0; i < nCells(); ++i) {
            values_.push_back(this->matrix().diag()[i]);
        }

        for (label i = 0; i < nNeighbours(); ++i) {
            values_.push_back(this->matrix().upper()[i]);
        }
    };

    void sort_host_matrix_data(const IOField<label> *sorting_idxs) const
    {
        std::vector<scalar> tmp_values(nElems());

        for (label i = 0; i < nElems(); i++) tmp_values[i] = values_[i];

        for (label i = 0; i < nElems(); i++) {
            label j = sorting_idxs->operator[](i);
            values_[i] = tmp_values[j];
        }
    };

    label nCells() const { return nCells_; };

    label nElems() const { return nElems_; };

    label nNeighbours() const { return nNeighbours_; };

    std::vector<scalar> &values() const { return values_; };

    std::vector<label> &col_idxs() const { return col_idxs_; };

    std::vector<label> &row_idxs() const { return row_idxs_; };
};

void export_system(const word fieldName, const mtx *A, const vec *x,
                   const vec *b, const word time);


// Logs the number of iteration executed
struct IterationLogger : gko::log::Logger {
    void on_iteration_complete(const gko::LinOp *,
                               const gko::size_type &num_iterations,
                               const gko::LinOp *residual,
                               const gko::LinOp *res_norm,
                               const gko::LinOp *) const override
    {
        this->num_iters = num_iterations;
    }

    IterationLogger(std::shared_ptr<const gko::Executor> exec)
        : exec(exec),
          gko::log::Logger(exec, gko::log::Logger::iteration_complete_mask)
    {}

    gko::size_type get_iters() { return num_iters; }

private:
    std::shared_ptr<const gko::Executor> exec;
    mutable gko::size_type num_iters{0};
};


template <class MatrixType, class SolverFactory>
class lduLduBase : public HostMatrix<MatrixType>,
                   public SolverFactory,
                   public StoppingCriterion,
                   public IOSortingIdxHandler,
                   public IOGKOMatrixHandler,
                   public IOPreconditioner {
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
          IOPreconditioner(matrix.mesh().thisDb(), solverControls, fieldName)
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
          IOPreconditioner(matrix.mesh().thisDb(), solverControls, fieldName)
    {
        init_base();
    }

    void init_base()
    {
        // if sys_matrix is not stored updating is neccesary
        // initially
        bool stored = get_sys_matrix_stored();
        if (!stored) {
            SIMPLE_TIME(init_host_sparsity_pattern,
                        this->init_host_sparsity_pattern();)
            std::cout << " matrix not stored update host matrix " << std::endl;
            SIMPLE_TIME(update_host_mtx, this->update_host_matrix_data();)
        } else {
            // if sys_matrix is  stored updating is only neccesary
            // when requested explictly
            if (get_update_sys_matrix()) {
                std::cout
                    << " matrix is stored and  update of host matrix requested"
                    << std::endl;
                SIMPLE_TIME(exp_update_host_mtx,
                            this->update_host_matrix_data();)
            }
        }

        // after updating the host matrix the host matrix needs to be sorted
        if (!get_is_sorted()) {
            std::cout << "matrix is not yet sorted, compute sorting idxs "
                      << std::endl;
            SIMPLE_TIME(compute_sort, this->compute_sorting_idxs(
                                          this->row_idxs(), this->col_idxs(),
                                          this->nCells());)
        }

        if (!stored && get_sort()) {
            std::cout << "matrix is not yet sorted,  sorting host matrix "
                      << std::endl;
            SIMPLE_TIME(
                sort_host_mtx_sparsity_pattern,
                this->sort_host_matrix_sparsity_pattern(get_sorting_idxs());)
            SIMPLE_TIME(sort_host_mtx,
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

        criterion_vec.push_back(build_stopping_criterion(device_exec, 1.0));

        // Generate solver
        auto solver_gen = this->create_solver(device_exec, criterion_vec);

        std::shared_ptr<mtx> gkomatrix = get_gkomatrix();

        auto solver = solver_gen->generate(gko::share(gkomatrix));

        for (int i = 0; i < 3; i++) {
            SIMPLE_TIME(solve, solver->apply(gko::lend(b[i]), gko::lend(x[i]));)
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
        this->matrix().Amul(wA, psi, this->interfaceBouCoeffs_,
                            this->interfaces_, cmpt);
        scalar norm_factor = this->normFactor(psi, source, wA, pA);

        auto source_view = val_array::view(ref_exec(), this->nCells(),
                                           const_cast<scalar *>(&source[0]));

        std::vector<val_array> source_views{source_view};
        auto b = vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1),
                             source_views[0], 1);

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
        SIMPLE_TIME(init_preconditioner,
                    this->init_preconditioner(this->matrix().mesh().thisDb(),
                                              gkomatrix, device_exec);)

        auto bj_generated = this->get_preconditioner();
        auto solver_gen =
            this->create_solver(device_exec, criterion_vec, bj_generated);

        // Instantiate a ResidualLogger logger.
        auto logger = std::make_shared<IterationLogger>(device_exec);

        // Add the previously created logger to the solver factory. The
        // logger will be automatically propagated to all solvers created
        // from this factory.
        solver_gen->add_logger(logger);

        // auto b_clone = gko::clone(b);


        if (get_export())
            export_system(this->fieldName(), gko::lend(gkomatrix),
                          gko::lend(x[0]), gko::lend(b),
                          this->matrix().mesh().thisDb().time().timeName());


        solverPerf.initialResidual() = this->get_init_res_norm();

        auto solver = solver_gen->generate(gko::share(gkomatrix));

        // Solve system
        SIMPLE_TIME(solve, solver->apply(gko::lend(b), gko::lend(x[0]));)
        std::cout << " copy back" << std::endl;

        this->copy_result_back(psi, this->nCells());

        // b_clone->copy_from(gko::lend(b));

        solverPerf.finalResidual() = this->get_res_norm();

        std::cout << " get iters " << std::endl;
        solverPerf.nIterations() = logger->get_iters();

        return solverPerf;
    };
};


}  // namespace Foam

#endif
