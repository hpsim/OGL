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

#ifndef GKOBase_H
#define GKOBase_H

#define SIMPLE_TIME(NAME, F)                                            \
    auto start_##NAME = std::chrono::steady_clock::now();               \
    F auto end_##NAME = std::chrono::steady_clock::now();               \
    std::cout << "Gingko " #NAME " : "                                  \
              << std::chrono::duration_cast<std::chrono::milliseconds>( \
                     end_##NAME - start_##NAME)                         \
                     .count()                                           \
              << " ms\n";


#include "lduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

// Logs the number of iteration executed
struct IterationLogger : gko::log::Logger {
    void on_iteration_complete(const gko::LinOp *,
                               const gko::size_type &num_iterations,
                               const gko::LinOp *, const gko::LinOp *,
                               const gko::LinOp *) const override
    {
        this->num_iters = num_iterations;
    }

    IterationLogger(std::shared_ptr<const gko::Executor> exec)
        : gko::log::Logger(exec, gko::log::Logger::iteration_complete_mask)
    {}

    gko::size_type get_iters() { return num_iters; }

private:
    mutable gko::size_type num_iters{0};
};

/*---------------------------------------------------------------------------*\
                           Class GKOCG Declaration
\*---------------------------------------------------------------------------*/

template <class SolverFactory>
class GKOBaseSolver : public lduMatrix::solver, public SolverFactory {
public:
    // Some shortcuts
    using IndexType = label;
    using vec = gko::matrix::Dense<scalar>;
    using mtx = gko::matrix::Coo<scalar>;
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;
    using cg = gko::solver::Cg<scalar>;

private:
    // executor where Ginkgo will perform the computation
    std::shared_ptr<gko::Executor> exec_;

    // executor of the application
    std::shared_ptr<gko::Executor> app_exec_;

    label nCells_;

    label nNeighbours_;

    label nElems_;

    bool export_;

    // for now a copy of the OF matrix is stored
    // to keep values contiguous in memory
public:
    std::vector<scalar> sorting_idxs_;

    mutable std::vector<scalar> values_;

    mutable std::vector<label> col_idxs_;

    mutable std::vector<label> row_idxs_;


public:
    //- Construct from matrix components and solver controls
    GKOBaseSolver(const word &fieldName, const lduMatrix &matrix,
                  const FieldField<Field, scalar> &interfaceBouCoeffs,
                  const FieldField<Field, scalar> &interfaceIntCoeffs,
                  const lduInterfaceFieldPtrsList &interfaces,
                  const dictionary &solverControls)
        : lduMatrix::solver(fieldName, matrix, interfaceBouCoeffs,
                            interfaceIntCoeffs, interfaces, solverControls),
          SolverFactory(controlDict_)
    {
        Info << "GKOBase ctr" << endl;
        // create executors
        auto executor_string =
            controlDict_.lookupOrDefault("executor", word("reference"));
        auto app_executor_string =
            controlDict_.lookupOrDefault("app_executor", word("reference"));

        const auto omp = gko::OmpExecutor::create();

        std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
            exec_map{
                {"omp", [&omp] { return omp; }},
                {"cuda",
                 [&omp] { return gko::CudaExecutor::create(0, omp, true); }},
                {"hip",
                 [&omp] { return gko::HipExecutor::create(0, omp, true); }},
                {"reference",
                 [&omp] { return gko::ReferenceExecutor::create(); }}};

        exec_ = exec_map.at(executor_string)();

        if (executor_string == app_executor_string) {
            app_exec_ = exec_;
        } else {
            app_exec_ = exec_map.at(app_executor_string)();
        }

        // resize matrix
        nCells_ = matrix.diag().size();
        nNeighbours_ = matrix.lduAddr().upperAddr().size();
        nElems_ = nCells_ + 2 * nNeighbours_;

        export_ = controlDict_.lookupOrDefault<bool>("export", false);

        SIMPLE_TIME(update_gko_mtx, update_GKOMatrix();)

        SIMPLE_TIME(compute_sort, compute_sorting_idxs();)

        SIMPLE_TIME(sort_gko_mtx, sort_GKOMatrix ();)
    };

    void compute_sorting_idxs()
    {
        Info << "sorting idxs() " << endl;
        // initialize original index locations
        sorting_idxs_.resize(nElems_);
        iota(sorting_idxs_.begin(), sorting_idxs_.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        stable_sort(sorting_idxs_.begin(), sorting_idxs_.end(),
                    [this](size_t i1, size_t i2) {
                        return row_idxs_[i1] < row_idxs_[i2];
                    });
    };

    std::shared_ptr<gko::Executor> exec() const { return exec_; };

    std::shared_ptr<gko::Executor> app_exec() const { return app_exec_; };

    const label nCells() const { return nCells_; };

    const label nElems() const { return nElems_; };

    const std::vector<scalar> &vals() const { return values_; };

    const std::vector<label> &col_idx() const { return col_idxs_; };

    const std::vector<label> &row_idx() const { return row_idxs_; };

    void update_GKOMatrix() const
    {
        // reset vectors
        values_.resize(0);
        col_idxs_.resize(0);
        row_idxs_.resize(0);

        values_.reserve(nElems_);
        col_idxs_.reserve(nElems_);
        row_idxs_.reserve(nElems_);


        // fill vectors unsorted
        for (IndexType i = 0; i < nNeighbours_; ++i) {
            values_.push_back(matrix().lower()[i]);
            row_idxs_.push_back(matrix().lduAddr().lowerAddr()[i]);
            col_idxs_.push_back(matrix().lduAddr().upperAddr()[i]);
        }

        for (IndexType i = 0; i < nCells_; ++i) {
            values_.push_back(matrix().diag()[i]);
            col_idxs_.push_back(i);
            row_idxs_.push_back(i);
        }

        for (IndexType i = 0; i < nNeighbours_; ++i) {
            values_.push_back(matrix().upper()[i]);
            row_idxs_.push_back(matrix().lduAddr().upperAddr()[i]);
            col_idxs_.push_back(matrix().lduAddr().lowerAddr()[i]);
        }
    };

    void sort_GKOMatrix() const
    {
        Info << "sorting matrix() " << endl;
        std::vector<scalar> tmp_values(nElems_);
        std::vector<scalar> tmp_col_idxs(nElems_);
        std::vector<scalar> tmp_row_idxs(nElems_);

        for (label i = 0; i < nElems_; i++) tmp_values[i] = values_[i];
        for (label i = 0; i < nElems_; i++) tmp_col_idxs[i] = col_idxs_[i];
        for (label i = 0; i < nElems_; i++) tmp_row_idxs[i] = row_idxs_[i];

        // TODO parallelise
        for (label i = 0; i < nElems_; i++) {
            label j = sorting_idxs_[i];
            values_[i] = tmp_values[j];
            col_idxs_[i] = tmp_col_idxs[j];
            row_idxs_[i] = tmp_row_idxs[j];
            // std::cout << "elem " << i << ": " << row_idxs_[i] << " " <<
            // col_idxs_[i] << std::endl;
        }
    };

    scalar compute_residual(mtx *const A, vec *const x, vec *const b) const
    {
        auto one = gko::initialize<vec>({1.0}, exec());
        auto neg_one = gko::initialize<vec>({-1.0}, exec());
        auto res = gko::initialize<vec>({0.0}, exec());
        A->apply(lend(one), lend(x), lend(neg_one), lend(b));
        b->compute_norm2(lend(res));

        return res->at(0);
    };

    virtual solverPerformance solve_impl(word typeName, scalarField &psi,
                                         const scalarField &source,
                                         const direction cmpt = 0) const
    {
        // --- Setup class containing solver performance data
        solverPerformance solverPerf(
            lduMatrix::preconditioner::getName(controlDict_) + typeName,
            fieldName_);

        auto psi_view = val_array::view(app_exec(), nCells(),
                                        const_cast<scalar *>(&psi[0]));
        auto source_view = val_array::view(app_exec(), nCells(),
                                           const_cast<scalar *>(&source[0]));

        auto b =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), source_view, 1);

        auto x = vec::create(app_exec(), gko::dim<2>(nCells(), 1), psi_view, 1);

        // Generate solver
        auto solver_gen = this->create_solver(exec());

        // Instantiate a ResidualLogger logger.
        auto logger = std::make_shared<IterationLogger>(exec());

        // Add the previously created logger to the solver factory. The logger
        // will be automatically propagated to all solvers created from this
        // factory.
        solver_gen->add_logger(logger);

        auto gkomatrix = gko::give(
            mtx::create(exec(), gko::dim<2>(nCells()),
                        val_array::view(app_exec(), nElems(), &values_[0]),
                        idx_array::view(app_exec(), nElems(), &col_idxs_[0]),
                        idx_array::view(app_exec(), nElems(), &row_idxs_[0])));

        // TODO refactor this to separate function
        if (export_) {
            std::string fn_mtx{fieldName() + "_A.mtx"};
            std::ofstream stream{fn_mtx};
            std::cerr << "Writing " << fn_mtx << std::endl;
            gko::write(stream, gkomatrix.get(), gko::layout_type::coordinate);

            std::string fn_b{fieldName() + "_b.mtx"};
            std::ofstream stream_b{fn_b};
            std::cerr << "Writing " << fn_b << std::endl;
            gko::write(stream_b, b.get());

            std::string fn_x{fieldName() + "_x.mtx"};
            std::ofstream stream_x{fn_x};
            std::cerr << "Writing " << fn_x << std::endl;
            gko::write(stream_x, x.get());
        }

        auto b_clone = gko::clone(b);

        solverPerf.initialResidual() = compute_residual(
            gko::lend(gkomatrix), gko::lend(x), gko::lend(b_clone));

        auto solver = solver_gen->generate(gko::give(gkomatrix));

        // Solve system
        SIMPLE_TIME(solve, solver->apply(gko::lend(b), gko::lend(x));)

        auto x_view = val_array::view(app_exec(), nCells(), x->get_values());
        auto x_clone =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), x_view, 1);
        psi_view = x_view;
        b_clone->copy_from(gko::lend(b));
        // auto ginkgo_host_matrix = gko::give(
        //     mtx::create(app_exec(), gko::dim<2>(nCells(), nCells()),
        //                 val_array::view(app_exec(), nElems(), &values_[0]),
        //                 idx_array::view(app_exec(), nElems(), &col_idxs_[0]),
        //                 idx_array::view(app_exec(), nElems(), &row_idxs_[0])));

        // solverPerf.finalResidual() =
        //     compute_residual(gko::lend(ginkgo_host_matrix), gko::lend(x_clone),
        //                      gko::lend(b_clone));

        solverPerf.nIterations() = logger->get_iters();

        return solverPerf;
    };
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

}  // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
