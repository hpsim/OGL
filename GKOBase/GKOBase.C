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
#include "GKOBase.H"

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
    auto app_executor_string = controlDict_.lookupOrDefault("app_executor", word("reference"));

    const auto omp = gko::OmpExecutor::create();

    std::map<std::string, std::shared_ptr<gko::Executor>> exec_map{
        {"omp", omp},
        {"cuda", gko::CudaExecutor::create(0, omp, true)},
        {"hip", gko::HipExecutor::create(0, omp, true)},
        {"reference", gko::ReferenceExecutor::create()}};

    exec_ = exec_map.at(executor_string);
    app_exec_ = exec_map.at(app_executor_string);


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
        // std::cout << "elem " << i << ": " << row_idxs_[i] << " " << col_idxs_[i] << std::endl;
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
    for (IndexType i = 0; i < nNeighbours_;  ++i) {
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



}


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

// ************************************************************************* //
