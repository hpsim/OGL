/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>
#include "GKOCG.H"

Foam::GKOBaseSolver::GKOBaseSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
    )
    :
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
        )
{
    // create executors
    auto executor_string = controlDict_.lookupOrDefault("executor", word("reference"));

    const auto omp = gko::OmpExecutor::create();

    std::map<std::string, std::shared_ptr<gko::Executor>> exec_map{
        {"omp", omp},
        {"cuda", gko::CudaExecutor::create(0, omp, true)},
        {"hip", gko::HipExecutor::create(0, omp, true)},
        {"reference", gko::ReferenceExecutor::create()}};

    exec_ = exec_map.at(executor_string);

    app_exec_= exec_map["omp"];

    // resize matrix
    nCells_ = matrix.diag().size();
    nNeighbours_ = matrix.lduAddr().upperAddr().size();
    nElems_ = nCells_ + 2*nNeighbours_;

    update_GKOMatrix();

    compute_sorting_idxs();

}


void Foam::GKOBaseSolver::compute_sorting_idxs() {

  // initialize original index locations
  sorting_idxs_.resize(nElems_);
  iota(sorting_idxs_.begin(), sorting_idxs_.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(sorting_idxs_.begin(), sorting_idxs_.end(),
       [this](size_t i1, size_t i2) {return row_idxs_[i1] < row_idxs_[i2];});
}

void Foam::GKOBaseSolver::sort_GKOMatrix() const {

    std::vector<scalar> tmp_values(nElems_);
    std::vector<scalar> tmp_col_idxs(nElems_);
    std::vector<scalar> tmp_row_idxs(nElems_);

    for(label i=0;i<nElems_;i++) tmp_values[i] = values_[i];
    for(label i=0;i<nElems_;i++) tmp_col_idxs[i] = col_idxs_[i];
    for(label i=0;i<nElems_;i++) tmp_row_idxs[i] = row_idxs_[i];

    for(label i=0;i<nElems_;i++) {
        label j = sorting_idxs_[i];
        values_[i] = tmp_values[j];
        col_idxs_[i] = tmp_col_idxs[j];
        row_idxs_[i] = tmp_row_idxs[j];
    }
}


void Foam::GKOBaseSolver::update_GKOMatrix() const {

    // reset vectors
    values_.resize(0);
    col_idxs_.resize(0);
    row_idxs_.resize(0);

    values_.reserve(nElems_);
    col_idxs_.reserve(nElems_);
    row_idxs_.reserve(nElems_);


    // fill vectors unsorted
    for (IndexType i = 0; i < nCells_; ++i) {
        values_.push_back(matrix().diag()[i]);
        col_idxs_.push_back(i);
        row_idxs_.push_back(i);
    }

    for (IndexType i = 0; i < nNeighbours_;  ++i) {
        values_.push_back(matrix().lower()[i]);
        row_idxs_.push_back(matrix().lduAddr().lowerAddr()[i]);
        col_idxs_.push_back(matrix().lduAddr().upperAddr()[i]);
    }

    for (IndexType i = 0; i < nNeighbours_; ++i) {
        values_.push_back(matrix().upper()[i]);
        row_idxs_.push_back(matrix().lduAddr().upperAddr()[i]);
        col_idxs_.push_back(matrix().lduAddr().lowerAddr()[i]);
    }



}

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

    gko::size_type get_iters() {return num_iters;}

private:
    mutable gko::size_type num_iters{0};
};


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(GKOCG, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<GKOCG>
        addGKOCGSymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::GKOCG::GKOCG
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    GKOBaseSolver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{



}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::GKOCG::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    auto b = vec::create(
        exec(),
        gko::dim<2>(nCells(), 1),
        val_array::view(exec(), nCells(), const_cast<scalar *>(&source[0])),
        1);

    auto x = vec::create(
        exec(),
        gko::dim<2>(nCells(), 1),
        val_array::view(exec(), nCells(), &psi[0]),
        1);

    update_GKOMatrix();
    sort_GKOMatrix();

    // Generate solver
    auto solver_gen =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(maxIter_).on(exec()),
                    gko::stop::ResidualNormReduction<scalar>::build()
                    .with_reduction_factor(tolerance_)
                .on(exec()))
        .on(exec());

    // Instantiate a ResidualLogger logger.
    auto logger = std::make_shared<IterationLogger>(exec());

    // Add the previously created logger to the solver factory. The logger will
    // be automatically propagated to all solvers created from this factory.
    solver_gen->add_logger(logger);

    auto gkomatrix = gko::give(mtx::create(exec(),
        gko::dim<2>(nCells()),
        val_array::view(exec(), nElems(), &values_[0]),
        idx_array::view(exec(), nElems(), &col_idxs_[0]),
        idx_array::view(exec(), nElems(), &row_idxs_[0])));

    auto solver = solver_gen->generate(gko::give(gkomatrix));

    // Solve system
    solver->apply(gko::lend(b), gko::lend(x));

    auto one = gko::initialize<vec>({1.0}, exec());
    auto neg_one = gko::initialize<vec>({-1.0}, exec());
    auto res = gko::initialize<vec>({0.0}, exec());
    b->compute_norm2(lend(res));

    solverPerf.initialResidual() = 1.0;
    solverPerf.finalResidual() =  1.0;
    solverPerf.nIterations() =  logger->get_iters();

    return solverPerf;
}


// ************************************************************************* //
