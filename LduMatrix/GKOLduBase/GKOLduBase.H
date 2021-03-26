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

#ifndef GKO_LDUBase_H
#define GKO_LDUBase_H

#define SIMPLE_TIME(NAME, F)                                            \
    auto start_##NAME = std::chrono::steady_clock::now();               \
    F auto end_##NAME = std::chrono::steady_clock::now();               \
    std::cout << "Gingko " #NAME " : "                                  \
              << std::chrono::duration_cast<std::chrono::milliseconds>( \
                     end_##NAME - start_##NAME)                         \
                     .count()                                           \
              << " ms\n";

#include "LduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

/*---------------------------------------------------------------------------*\
                           Class GKOCG Declaration
\*---------------------------------------------------------------------------*/

template <class Type, class LDUType, class SolverFactory>
class GKOLduBaseSolver : public LduMatrix<Type, LDUType, LDUType>::solver,
                         public SolverFactory {
public:
    // Some shortcuts
    using IndexType = label;
    using vec = gko::matrix::Dense<scalar>;
    using mtx = gko::matrix::Coo<scalar>;
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;
    using cg = gko::solver::Cg<scalar>;

private:
    const label maxIterI_;

    // TODO implement min iter
    const label minIterI_;

    const scalar toleranceI_;

    const scalar relTol_;

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
    GKOLduBaseSolver(const word &fieldName,
                     const LduMatrix<Type, LDUType, LDUType> &matrix,
                     const dictionary &solverControls)
        : LduMatrix<Type, LDUType, LDUType>::solver(fieldName, matrix,
                                                    solverControls),
          SolverFactory(this->controlDict_),
          maxIterI_(this->controlDict_.lookupOrDefault("maxIter", label(1000))),
          minIterI_(this->controlDict_.lookupOrDefault("minIter", label(0))),
          toleranceI_(
              this->controlDict_.lookupOrDefault("tolerance", scalar(1e-6))),
          relTol_(this->controlDict_.lookupOrDefault("relTol", scalar(1e-6)))
    {
        // create executors
        auto executor_string =
            this->controlDict_.lookupOrDefault("executor", word("reference"));
        auto app_executor_string = this->controlDict_.lookupOrDefault(
            "app_executor", word("reference"));


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

        SIMPLE_TIME(update_gko_mtx, update_GKOMatrix();)

        // SIMPLE_TIME(compute_sort, compute_sorting_idxs();)

        // SIMPLE_TIME(sort_gko_mtx, sort_GKOMatrix();)
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
            values_.push_back(this->matrix_.lower()[i]);
            row_idxs_.push_back(this->matrix_.lduAddr().lowerAddr()[i]);
            col_idxs_.push_back(this->matrix_.lduAddr().upperAddr()[i]);
        }

        for (IndexType i = 0; i < nCells_; ++i) {
            values_.push_back(this->matrix_.diag()[i]);

            col_idxs_.push_back(i);
            row_idxs_.push_back(i);
        }

        for (IndexType i = 0; i < nNeighbours_; ++i) {
            values_.push_back(this->matrix_.upper()[i]);

            row_idxs_.push_back(this->matrix_.lduAddr().upperAddr()[i]);
            col_idxs_.push_back(this->matrix_.lduAddr().lowerAddr()[i]);
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

    virtual SolverPerformance<Type> solve_impl(Field<Type> &psi) const
    {
        // --- Setup class containing solver performance data
        // Implement
        word preconditionerName(this->controlDict_.lookup("preconditioner"));

        SolverPerformance<Type> solverPerf(preconditionerName,
                                           this->fieldName_);

        auto source_view_x = val_array::view(
            app_exec(), 3 * nCells(),
            const_cast<scalar *>(&this->matrix_.source()[0].x()));
        auto source_view_y = val_array::view(
            app_exec(), 3 * nCells(),
            const_cast<scalar *>(&this->matrix_.source()[0].y()));
        auto source_view_z = val_array::view(
            app_exec(), 3 * nCells(),
            const_cast<scalar *>(&this->matrix_.source()[0].z()));

        auto b_x =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), source_view_x, 3);
        auto b_y =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), source_view_y, 3);
        auto b_z =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), source_view_z, 3);

        auto psi_view_x = val_array::view(app_exec(), 3 * nCells(),
                                          const_cast<scalar *>(&psi[0].x()));
        auto psi_view_y = val_array::view(app_exec(), 3 * nCells(),
                                          const_cast<scalar *>(&psi[0].y()));
        auto psi_view_z = val_array::view(app_exec(), 3 * nCells(),
                                          const_cast<scalar *>(&psi[0].z()));

        auto x =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), psi_view_x, 3);
        auto y =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), psi_view_y, 3);
        auto z =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), psi_view_z, 3);


        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec{};

        auto tolerance_stop = gko::stop::AbsoluteResidualNorm<scalar>::build()
                                  .with_tolerance(toleranceI_)
                                  .on(exec());

        // NOTE currently minIter
        // TODO move to base class
        if (minIterI_ == 0) {
            auto max_iter_stop =
                gko::stop::Iteration::build().with_max_iters(minIterI_).on(
                    exec());
            criterion_vec.push_back(gko::share(tolerance_stop));
            criterion_vec.push_back(gko::share(max_iter_stop));
        } else {
            auto max_iter_stop =
                gko::stop::Iteration::build().with_max_iters(minIterI_).on(
                    exec());
            criterion_vec.push_back(gko::share(max_iter_stop));
        }

        // Generate solver
        auto solver_gen = this->create_solver(exec(), criterion_vec);

        auto gkomatrix = gko::give(
            mtx::create(exec(), gko::dim<2>(nCells()),
                        val_array::view(app_exec(), nElems(), &values_[0]),
                        idx_array::view(app_exec(), nElems(), &col_idxs_[0]),
                        idx_array::view(app_exec(), nElems(), &row_idxs_[0])));


        auto solver = solver_gen->generate(gko::give(gkomatrix));

        SIMPLE_TIME(solve_x, solver->apply(gko::lend(b_x), gko::lend(x));)
        SIMPLE_TIME(solve_y, solver->apply(gko::lend(b_y), gko::lend(y));)
        SIMPLE_TIME(solve_z, solver->apply(gko::lend(b_z), gko::lend(z));)

        auto x_view = val_array::view(app_exec(), nCells(), x->get_values());
        auto x_clone =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), x_view, 1);

        auto y_view = val_array::view(app_exec(), nCells(), y->get_values());
        auto y_clone =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), y_view, 1);

        auto z_view = val_array::view(app_exec(), nCells(), z->get_values());
        auto z_clone =
            vec::create(app_exec(), gko::dim<2>(nCells(), 1), z_view, 1);


        for (size_t i = 0; i < nCells(); i++) {
            psi[i].x() = x_clone->at(i);
            psi[i].y() = y_clone->at(i);
            psi[i].z() = z_clone->at(i);
        }

        return solverPerf;
    };
};
}  // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
