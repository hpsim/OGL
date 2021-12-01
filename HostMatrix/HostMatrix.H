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


Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    HostMatrix.H

\*---------------------------------------------------------------------------*/
#ifndef OGL_HostMatrix_INCLUDED_H
#define OGL_HostMatrix_INCLUDED_H
#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"

#include <vector>

namespace Foam {

template <class MatrixType>
class HostMatrix : public MatrixType::solver {
private:
    const label nCells_;

    const label nNeighbours_;

    const label nElems_;

    // TODO these are not neeeded as data members
    // since there only needed initially to create
    // the lduCsrMapping
    // const gkoGlobalIndex globalIndexCells_;

    // const gkoGlobalIndex globalIndexNeighbours_;

    const label nGlobalElems_;

    // TODO make this persistent
    // global index of the full matrix sparsity pattern
    const gkoGlobalIndex globalIndex_;

    // TODO rename to lduCsrIdxMapping
    const IOSortingIdxHandler sortingIdxHandler_;

    // TODO use devicePersistentFields here
    // TODO this will need a ->get_data() method
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
          nElems_(nCells_ + 2 * nNeighbours_),
          // globalIndexCells_(gkoGlobalIndex(nCells_)),
          // globalIndexNeighbours_(gkoGlobalIndex(nNeighbours_)),
          nGlobalElems_(2 * globalIndexNeighbours_.size() +
                        globalIndexCells_.size()),
          globalIndex_(gkoGlobalIndex(nElems_)),
          sortingIdxHandler_(
              IOSortingIdxHandler(matrix.mesh().thisDb(), nElems_)){};

    HostMatrix(const word &fieldName, const MatrixType &matrix,
               const dictionary &solverControls)
        : MatrixType::solver(fieldName, matrix, solverControls),
          nCells_(matrix.diag().size()),
          nNeighbours_(matrix.lduAddr().upperAddr().size()),
          nElems_(nCells_ + 2 * nNeighbours_),
          // globalIndexCells_(gkoGlobalIndex(nCells_)),
          // globalIndexNeighbours_(gkoGlobalIndex(nNeighbours_)),
          // TODO move to init_host_sparsity_pattern
          // nGlobalElems_(2 * globalIndexNeighbours_.size() +
          //               globalIndexCells_.size()),
          globalIndex_(gkoGlobalIndex(nElems_)),
          sortingIdxHandler_(
              IOSortingIdxHandler(matrix.mesh().thisDb(), nElems_)){};

    void init_host_sparsity_pattern() const
    {
        // Step 1 local ldu -> csr conversion
        // including local processor offset

        auto ref_exec = gko::ReferenceExecutor::create();


        auto lower_local = idx_array::view(
            ref_exec, nNeighbours_,
            const_cast<label *>(&this->matrix().lduAddr().lowerAddr()[0]));

        auto upper_local = idx_array::view(
            ref_exec, nNeighbours_,
            const_cast<label *>(&this->matrix().lduAddr().upperAddr()[0]));

        const auto lower = lower_local.get_const_data();
        const auto upper = upper_local.get_const_data();

        auto col_idxs_local = std::make_shared<idx_array>(ref_exec, nElems_);
        auto row_idxs_local = std::make_shared<idx_array>(ref_exec, nElems_);

        auto rows = row_idxs_local->get_data();
        auto cols = col_idxs_local->get_data();

        label element_ctr = 0;
        label upper_ctr = 0;
        label lower_ctr = 0;

        std::vector<std::vector<std::pair<label, label>>> lower_stack(
            nNeighbours_);


        auto sorting_idxs = sortingIdxHandler_.get_sorting_idxs();

        for (label row = 0; row < nCells_; row++) {
            // add lower elements
            // for now just scan till current upper ctr
            if (!lower_stack[row].empty()) {
                for (const auto &element : lower_stack[row]) {
                    rows[element_ctr] = globalIndexCells_.toGlobal(row);
                    cols[element_ctr] = element.second;
                    // lower_ctr doesnt correspond to same element as upper_ctr
                    sorting_idxs->operator[](element.first + nNeighbours_) =
                        element_ctr;

                    lower_ctr++;
                    element_ctr++;
                }
            }

            // add diagonal elemnts
            rows[element_ctr] = globalIndexCells_.toGlobal(row);
            cols[element_ctr] = globalIndexCells_.toGlobal(row);
            sorting_idxs->operator[](2 * nNeighbours_ + row) = element_ctr;

            element_ctr++;

            // add upper elemnts
            while (lower[upper_ctr] == row && upper_ctr < nNeighbours_) {
                label row_upper = globalIndexCells_.toGlobal(lower[upper_ctr]);
                label col_upper = globalIndexCells_.toGlobal(upper[upper_ctr]);
                rows[element_ctr] = row_upper;
                cols[element_ctr] = col_upper;

                // insert into lower_stack
                // find insert position
                // std::cout << " Proc " << Pstream::myProcNo() << "col_upper "
                //           << col_upper << "  row " << row << "  element_ctr"
                //           << element_ctr << endl;
                lower_stack[upper[upper_ctr]].emplace_back(upper_ctr,
                                                           row_upper);
                // std::cout << "done" << endl;
                sorting_idxs->operator[](upper_ctr) = element_ctr;

                element_ctr++;
                upper_ctr++;
            }
        }

        // Step 2. gather on master
        if (Pstream::parRun()) {
            const label nGlobalNeighbours = globalIndexNeighbours_.size();

            // TODO create without size
            auto col_idxs_global =
                std::make_shared<idx_array>(ref_exec, nGlobalElems_);

            // TODO create without size
            auto row_idxs_global =
                std::make_shared<idx_array>(ref_exec, nGlobalElems_);

            // gather lduAdressing
            // pass row and col idxs as reference instead of shared_ptr
            // row and col idxs are stored as shared_ptr to keep them in
            // objectReg
            // FIXME when col_idxs_ are properly wrapped gathering can
            // happen on demand in the wrapper
            globalIndex_.gather(*col_idxs_local.get(), *col_idxs_global.get());
            globalIndex_.gather(*row_idxs_local.get(), *row_idxs_global.get());

            if (Pstream::master()) {
                col_idxs_ = col_idxs_global;
                row_idxs_ = row_idxs_global;
            }

        } else {
            col_idxs_ = col_idxs_local;
            row_idxs_ = row_idxs_local;
        }
    }


    void update_host_matrix_data() const
    {
        auto ref_exec = gko::ReferenceExecutor::create();

        values_ = std::make_shared<val_array>(ref_exec, nElems_);

        auto sorting_idxs = sortingIdxHandler_.get_sorting_idxs();
        auto lower = this->matrix().lower();
        auto upper = this->matrix().upper();
        for (label i = 0; i < nNeighbours(); i++) {
            values_->get_data()[sorting_idxs->operator[](i)] = upper[i];
            values_->get_data()[sorting_idxs->operator[](i + nNeighbours_)] =
                lower[i];
        }

        auto diag = this->matrix().diag();
        for (label i = 0; i < nCells(); ++i) {
            values_
                ->get_data()[sorting_idxs->operator[](i + 2 * nNeighbours_)] =
                diag[i];
        }

        // Step 2. gather on master
        if (Pstream::parRun()) {
            auto values_global =
                std::make_shared<val_array>(ref_exec, nGlobalElems_);
            globalIndex_.gather(*values_.get(), *values_global.get());
            values_ = values_global;
        }
    };

    label nCells() const { return nCells_; };

    label nElems() const { return nElems_; };

    label nNeighbours() const { return nNeighbours_; };

    std::shared_ptr<val_array> values() const { return values_; };

    std::shared_ptr<idx_array> col_idxs() const { return col_idxs_; };

    std::shared_ptr<idx_array> row_idxs() const { return row_idxs_; };
};


}  // namespace Foam
#endif
