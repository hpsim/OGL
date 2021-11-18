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
#include "../IOGlobalIndex/globalIndex.H"
#include "../IOHandler/IOPreconditioner/IOPreconditioner.H"
#include "../IOHandler/IOSortingIdxHandler/IOSortingIdxHandler.H"
#include "../common/StoppingCriterion.H"

#include <ginkgo/ginkgo.hpp>

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

    const globalIndex globalIndexCells_;

    const globalIndex globalIndexNeighbours_;

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
          GlobalIndex(matrix.diag().size()) nCells_(matrix.diag().size()),
          nNeighbours_(matrix.lduAddr().upperAddr().size()),
          nElems_(nCells_ + 2 * nNeighbours_),
          globalIndexCells(globalIndex(nCells_)),
          globalIndexNeighbours(globalIndex(nNeighbours_)),
          {};

    HostMatrix(const word &fieldName, const MatrixType &matrix,
               const dictionary &solverControls)
        : MatrixType::solver(fieldName, matrix, solverControls),
          nCells_(matrix.diag().size()),
          nNeighbours_(matrix.lduAddr().upperAddr().size()),
          nElems_(nCells_ + 2 * nNeighbours_),
          globalIndexCells(globalIndex(nCells_)),
          globalIndexNeighbours(globalIndex(nNeighbours_)),
          {};

    void init_host_sparsity_pattern(IOField<label> *sorting_idxs) const
    {
        auto col_idxs_global =
            std::make_shared<idx_array>(gko::ReferenceExecutor::create(),
                                        globalIndexNeighbours_.globalSize());

        auto row_idxs_global =
            std::make_shared<idx_array>(gko::ReferenceExecutor::create(),
                                        globalIndexNeighbours_.globalSize());

        col_idxs_ = ( Pstream::parRun() ) ?
            std::make_shared<idx_array>(gko::ReferenceExecutor::create(),
                                        globIndexCells_.localSize())
            : col_idxs_global;

        row_idxs_ = ( Pstream::parRun() ) ?
            std::make_shared<idx_array>(gko::ReferenceExecutor::create(),
                                        globalIndexNeighbours_.localSize())
            : row_idxs_global;

        // gather lduAdressing

        // locally write ldu adressing in row column order
        auto lower = this->matrix().lduAddr().lowerAddr();
        auto upper = this->matrix().lduAddr().upperAddr();

        auto rows = row_idxs_->get_data();
        auto cols = col_idxs_->get_data();

        label element_ctr = 0;
        label upper_ctr = 0;
        label lower_ctr = 0;
        std::vector<std::vector<std::pair<label, label>>> lower_stack(
            nNeighbours_);
        for (label row = 0; row < nCells(); row++) {
            // add lower elements
            // for now just scan till current upper ctr
            if (!lower_stack[row].empty()) {
                for (const auto &element : lower_stack[row]) {
                    rows[element_ctr] = row;
                    cols[element_ctr] = element.second;
                    // lower_ctr doesnt correspond to same element as upper_ctr
                    sorting_idxs->operator[](element.first + nNeighbours_) =
                        element_ctr;

                    lower_ctr++;
                    element_ctr++;
                }
            }

            // add diagonal elemnts
            rows[element_ctr] = row;
            cols[element_ctr] = row;
            sorting_idxs->operator[](2 * nNeighbours_ + row) = element_ctr;

            element_ctr++;

            // add upper elemnts
            while (lower[upper_ctr] == row && upper_ctr < nNeighbours_) {
                label row = lower[upper_ctr];
                label col = upper[upper_ctr];
                rows[element_ctr] = row;
                cols[element_ctr] = col;

                // insert into lower_stack
                // find insert position
                lower_stack[col].emplace_back(upper_ctr, row);
                sorting_idxs->operator[](upper_ctr) = element_ctr;

                element_ctr++;
                upper_ctr++;
            }
        }
    }


    void update_host_matrix_data(const IOField<label> *sorting_idxs) const
    {
        label nCells_ = nCells();
        label nNeighbours_ = nNeighbours();

        values_ = std::make_shared<val_array>(gko::ReferenceExecutor::create(),
                                              nElems());

        auto values = values_->get_data();
        auto lower = this->matrix().lower();
        auto upper = this->matrix().upper();
        for (label i = 0; i < nNeighbours(); i++) {
            values[sorting_idxs->operator[](i)] = upper[i];
            values[sorting_idxs->operator[](i + nNeighbours_)] = lower[i];
        }

        auto diag = this->matrix().diag();
        for (label i = 0; i < nCells(); ++i) {
            values[sorting_idxs->operator[](i + 2 * nNeighbours_)] = diag[i];
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
