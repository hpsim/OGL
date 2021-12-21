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
#include "processorLduInterface.H"

#include <vector>
#include "../DevicePersistentArray/DevicePersistentArray.H"
#include "../IOGlobalIndex/gkoGlobalIndex.H"


namespace Foam {

template <class MatrixType>
class HostMatrixWrapper : public MatrixType::solver {
private:
    // dimensions of the local host matrix
    //
    const ExecutorHandler exec_;

    const label nCells_;

    const label nNeighbours_;

    const label nInterfaces_;

    const label nElems_;

    const gkoGlobalIndex global_cell_index_;
    // TODO make this persistent
    // global index of the full matrix sparsity pattern
    const gkoGlobalIndex global_matrix_index_;

    const gkoGlobalIndex global_values_index_;

    const label verbose_;

    // TODO
    mutable PersistentArray<label> ldu_csr_idx_mapping_;

    // TODO use devicePersistentFields here
    // TODO this will need a ->get_data() method
    mutable PersistentArray<scalar> values_;

    // NOTE DevicePersistentArray is used here
    // since at some point a global col_idxs row_idxs are neeeded
    // for parallel runs
    // TODO make shure the col_idxs_ arrays are not! stored since
    // they will be stored in the gko matrix
    mutable PersistentArray<label> col_idxs_;

    mutable PersistentArray<label> row_idxs_;


public:
    // segregated wrapper constructor
    HostMatrixWrapper(const objectRegistry &db, const MatrixType &matrix,
                      const FieldField<Field, scalar> &interfaceBouCoeffs,
                      const FieldField<Field, scalar> &interfaceIntCoeffs,
                      const lduInterfaceFieldPtrsList &interfaces,
                      const dictionary &solverControls, const word &fieldName)
        : MatrixType::solver(fieldName, matrix, interfaceBouCoeffs,
                             interfaceIntCoeffs, interfaces, solverControls),
          exec_{db, solverControls, fieldName},
          nCells_(matrix.diag().size()),
          nNeighbours_(matrix.lduAddr().upperAddr().size()),
          nInterfaces_(count_elements_on_interfaces(interfaces)),
          nElems_(nCells_ + 2 * nNeighbours_ + nInterfaces_),
          global_cell_index_{nCells_},
          global_matrix_index_{nElems_},
          global_values_index_{nElems_ - nInterfaces_},
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          ldu_csr_idx_mapping_{
              fieldName + "ldu_csr_map", db,       exec_,
              global_values_index_,      verbose_, false,
          },
          values_{fieldName + "_values", db,       exec_,
                  global_matrix_index_, verbose_, false},
          col_idxs_{fieldName + "_cols",   db,       exec_,
                    global_matrix_index_, verbose_, false},
          row_idxs_{fieldName + "_rows",   db,       exec_,
                    global_matrix_index_, verbose_, false}
    {
        if (!col_idxs_.get_stored() || col_idxs_.get_update()) {
            SIMPLE_TIME(true, init_host_sparsity_pattern,
                        init_host_sparsity_pattern(interfaces);)
        }
        if (!values_.get_stored() || values_.get_update()) {
            SIMPLE_TIME(true, update_host_matrix_data,
                        update_host_matrix_data(interfaces);)
        }
    }

    // coupled wrapper constructor
    HostMatrixWrapper(const objectRegistry &db, const MatrixType &matrix,
                      const dictionary &solverControls, const word &fieldName)
        : MatrixType::solver(fieldName, matrix, solverControls),
          exec_{db, solverControls, fieldName},
          nCells_(matrix.diag().size()),
          nNeighbours_(matrix.lduAddr().upperAddr().size()),
          nInterfaces_(0),
          nElems_(nCells_ + 2 * nNeighbours_),
          global_cell_index_{nCells_},
          global_matrix_index_{nElems_},
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          ldu_csr_idx_mapping_{fieldName + "ldu_csr_map", db,       exec_,
                               global_matrix_index_,      verbose_, false},
          values_{fieldName + "values", db,       exec_,
                  global_matrix_index_, verbose_, false},
          col_idxs_{fieldName + "cols",   db,       exec_,
                    global_matrix_index_, verbose_, false},
          row_idxs_{fieldName + "rows",   db,       exec_,
                    global_matrix_index_, verbose_, false}
    {
        // if (!col_idxs_.get_stored() || col_idxs_.get_update()) {
        //     SIMPLE_TIME(true, init_host_sparsity_pattern,
        //                 init_host_sparsity_pattern();)
        // }

        // if (!values_.get_stored() || values_.get_update()) {
        //     SIMPLE_TIME(true, update_host_matrix_data,
        //                 update_host_matrix_data();)
        // }
    }

    label count_elements_on_interfaces(
        const lduInterfaceFieldPtrsList &interfaces_) const
    {
        label ctr{0};
        for (int i = 0; i < interfaces_.size(); i++) {
            if (interfaces_.operator()(i) == nullptr) {
                continue;
            }
            const auto iface{interfaces_.operator()(i)};
            ctr += iface->interface().faceCells().size();
        }
        return ctr;
    }

    void insert_interface_coeffs(const lduInterfaceFieldPtrsList &interfaces,
                                 int *rows, int *cols, label row,
                                 label &element_ctr, const bool upper) const
    {
        for (int i = 0; i < interfaces.size(); i++) {
            if (interfaces.operator()(i) == nullptr) {
                continue;
            }
            const auto iface{interfaces.operator()(i)};

            if (isA<processorLduInterface>(iface->interface())) {
                const processorLduInterface &pldui =
                    refCast<const processorLduInterface>(iface->interface());

                // if rank of corresponding processor is greater
                // then own processor idx are not on lower matrix row
                if (upper) {
                    if (pldui.neighbProcNo() > Pstream::myProcNo()) {
                        continue;
                    }
                } else {
                    if (pldui.neighbProcNo() < Pstream::myProcNo()) {
                        continue;
                    }
                }

		std::cout << "!!!!!!! send \n" ;
                pldui.send(Pstream::commsTypes::blocking,
                           iface->interface().faceCells());
		std::cout << "!!!!!!! send done \n" ;

                auto otherSide_tmp =
                    pldui.receive<label>(Pstream::commsTypes::blocking,
                                         iface->interface().faceCells().size());
		std::cout << "!!!!!!! receive done \n" ;



                // check if current cell is on the current patch
                // NOTE cells can be several times on same patch
                for (label cellI = 0;
                     cellI < iface->interface().faceCells().size(); cellI++) {
                    label other_side_cellID = otherSide_tmp()[cellI];
		const label other_side_global_cellID = global_cell_index_.toGlobal(
                            pldui.neighbProcNo(), other_side_cellID);
		    //std::cout << " other_side_cellID " << other_side_cellID 
	//		    <<  " other_side_global_cellID " <<  other_side_global_cellID << "\n";

                    if (iface->interface().faceCells()[cellI] == row) {
                        rows[element_ctr] = global_cell_index_.toGlobal(row);
                        cols[element_ctr] = other_side_global_cellID;
                        element_ctr++;
                    }
                }
            }
        }
    }

    void init_host_sparsity_pattern(
        const lduInterfaceFieldPtrsList &interfaces) const
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

        auto rows = row_idxs_.get_data();  // row_idxs_local->get_data();
        auto cols = col_idxs_.get_data();  // col_idxs_local->get_data();

        label element_ctr = 0;
        label upper_ctr = 0;
        label lower_ctr = 0;

        std::vector<std::vector<std::pair<label, label>>> lower_stack(
            nNeighbours_);


        const auto sorting_idxs = ldu_csr_idx_mapping_.get_data();

        for (label row = 0; row < nCells_; row++) {
            // check for lower idxs
            insert_interface_coeffs(interfaces, rows, cols, row, element_ctr,
                                    true);

            // add lower elements
            // for now just scan till current upper ctr
            if (!lower_stack[row].empty()) {
                for (const auto &element : lower_stack[row]) {
                    // TODO
                    rows[element_ctr] = global_cell_index_.toGlobal(row);
                    cols[element_ctr] = element.second;
                    // lower_ctr doesnt correspond to same element as
                    // upper_ctr
                    sorting_idxs[element.first + nNeighbours_] = element_ctr;

                    lower_ctr++;
                    element_ctr++;
                }
            }

            // add diagonal elemnts
            rows[element_ctr] = global_cell_index_.toGlobal(row);
            cols[element_ctr] = global_cell_index_.toGlobal(row);
            sorting_idxs[2 * nNeighbours_ + row] = element_ctr;

            element_ctr++;

            // add upper elemnts
            while (lower[upper_ctr] == row && upper_ctr < nNeighbours_) {
                label row_upper = global_cell_index_.toGlobal(lower[upper_ctr]);
                label col_upper = global_cell_index_.toGlobal(upper[upper_ctr]);
                rows[element_ctr] = row_upper;
                cols[element_ctr] = col_upper;

                // insert into lower_stack
                // find insert position
                // std::cout << " Proc " << Pstream::myProcNo() <<
                // "col_upper "
                //           << col_upper << "  row " << row << "
                //           element_ctr"
                //           << element_ctr << endl;
                lower_stack[upper[upper_ctr]].emplace_back(upper_ctr,
                                                           row_upper);
                // std::cout << "done" << endl;
                sorting_idxs[upper_ctr] = element_ctr;

                element_ctr++;
                upper_ctr++;
            }
            insert_interface_coeffs(interfaces, rows, cols, row, element_ctr,
                                    false);
        }
    }

    bool get_verbose() const { return verbose_; }

    void update_host_matrix_data(
        const lduInterfaceFieldPtrsList &interfaces) const
    {
        auto ref_exec = gko::ReferenceExecutor::create();
        // TODO create in ctr
        // as devicePersistent so that we can reuse the memory

        auto values = values_.get_data();

        const auto sorting_idxs = ldu_csr_idx_mapping_.get_const_data();
        auto lower = this->matrix().lower();
        auto upper = this->matrix().upper();
        for (label i = 0; i < nNeighbours(); i++) {
            values[sorting_idxs[i]] = upper[i];
            values[sorting_idxs[i + nNeighbours_]] = lower[i];
        }

        auto diag = this->matrix().diag();
        for (label i = 0; i < local_nCells(); ++i) {
            values[sorting_idxs[i + 2 * nNeighbours_]] = diag[i];
        }
    };

    label local_nCells() const { return nCells_; }

    label nCells() const { return global_cell_index_.size(); }

    label nElems() const { return nElems_; }

    label nNeighbours() const { return nNeighbours_; }

    const ExecutorHandler &get_exec_handler() const { return exec_; }

    const gkoGlobalIndex &get_global_cell_index() const
    {
        return global_cell_index_;
    }


    PersistentArray<scalar> &get_values() const { return values_; };

    PersistentArray<label> &get_col_idxs() const { return col_idxs_; };

    PersistentArray<label> &get_row_idxs() const { return row_idxs_; };
};


}  // namespace Foam
#endif
