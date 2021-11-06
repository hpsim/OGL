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


Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    common.H

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

namespace Foam {
using mtx = gko::matrix::Csr<scalar>;
using vec = gko::matrix::Dense<scalar>;

#define SIMPLE_LOG(VERBOSE, MSG)                       \
    if (VERBOSE) {                                     \
        std::cout << "[OGL LOG] " << MSG << std::endl; \
    }

#define SIMPLE_TIME(VERBOSE, NAME, F)                                       \
    auto start_##NAME = std::chrono::steady_clock::now();                   \
    F auto end_##NAME = std::chrono::steady_clock::now();                   \
    if (VERBOSE) {                                                          \
        std::cout << "[OGL LOG] " #NAME " : "                               \
                  << std::chrono::duration_cast<std::chrono::microseconds>( \
                         end_##NAME - start_##NAME)                         \
                         .count()                                           \
                  << " mu s\n";                                             \
    }


template <class MatrixType>
class HostMatrix : public MatrixType::solver {
private:
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;

    const label nCells_;

    const label nNeighbours_;

    const label nElems_;

    mutable std::shared_ptr<val_array> values_;

    mutable std::shared_ptr<idx_array> col_idxs_;

    mutable std::shared_ptr<idx_array> row_idxs_;

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
        col_idxs_ = std::make_shared<idx_array>(
            gko::ReferenceExecutor::create(), nElems());

        row_idxs_ = std::make_shared<idx_array>(
            gko::ReferenceExecutor::create(), nElems());

        // fill vectors
        // unsorted
        for (label i = 0; i < nNeighbours(); ++i) {
            row_idxs_->get_data()[i] = this->matrix().lduAddr().lowerAddr()[i];
            col_idxs_->get_data()[i] = this->matrix().lduAddr().upperAddr()[i];
        }

        for (label i = nNeighbours(); i < nCells() + nNeighbours(); ++i) {
            col_idxs_->get_data()[i] = i;
            row_idxs_->get_data()[i] = i;
        }

        for (label i = nCells() + nNeighbours(); i < nElems(); ++i) {
            row_idxs_->get_data()[i] = this->matrix().lduAddr().upperAddr()[i];
            col_idxs_->get_data()[i] = this->matrix().lduAddr().lowerAddr()[i];
        }
    }

    void sort_host_matrix_sparsity_pattern(
        const IOField<label> *sorting_idxs) const
    {
        idx_array tmp_col_idxs{col_idxs_};
        idx_array tmp_row_idxs{row_idxs_};

        for (label i = 0; i < nElems(); i++) {
            label j = sorting_idxs->operator[](i);
            col_idxs_->get_data()[j] = tmp_col_idxs.get_data()[i];
            row_idxs_->get_data()[j] = tmp_row_idxs.get_data()[i];
        }
    };


    void update_host_matrix_data(const IOField<label> *sorting_idxs) const
    {
        label nCells_ = nCells();
        label nNeighbours_ = nNeighbours();

        values_ = std::make_shared<val_array>(gko::ReferenceExecutor::create(),
                                              nElems());

        for (label i = 0; i < nNeighbours(); ++i) {
            values_->get_data()[sorting_idxs->operator[](i)] =
                this->matrix().lower()[i];
            values_->get_data()[sorting_idxs->operator[](
                i + nCells_ + nNeighbours_)] = this->matrix().upper()[i];
        }

        // TODO does this even need to look up the sorting?
        // d values could be added to l or u values
        for (label i = 0; i < nCells(); ++i) {
            values_->get_data()[sorting_idxs->operator[](i + nNeighbours_)] =
                this->matrix().diag()[i];
        }
    };

    label nCells() const { return nCells_; };

    label nElems() const { return nElems_; };

    label nNeighbours() const { return nNeighbours_; };

    std::shared_ptr<val_array> values() const { return values_; };

    std::shared_ptr<idx_array> col_idxs() const { return col_idxs_; };

    std::shared_ptr<idx_array> row_idxs() const { return row_idxs_; };
};

void export_system(const word fieldName, const mtx *A, const vec *x,
                   const vec *b, const word time);

void export_vec(const word fieldName, const vec *x, const word time);

void set_solve_prev_iters(const word sys_matrix_name, const objectRegistry &db,
                          label prev_solve_iters);

label get_solve_prev_iters(const word sys_matrix_name,
                           const objectRegistry &db);
}  // namespace Foam

#endif
