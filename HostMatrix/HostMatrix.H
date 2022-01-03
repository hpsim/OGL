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
#include "../common/common.H"


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

    const gkoGlobalIndex global_interface_index_;

    const label verbose_;

    mutable PersistentArray<label> ldu_csr_idx_mapping_;

    mutable PersistentArray<label> ldu_csr_idx_interface_mapping_;

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
          global_interface_index_{nInterfaces_},
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          ldu_csr_idx_mapping_{
              fieldName + "_ldu_csr_map", db,       exec_,
              global_values_index_,       verbose_, false,
          },
          ldu_csr_idx_interface_mapping_{
              fieldName + "_ldu_csr_interface_map",
              db,
              exec_,
              global_interface_index_,
              verbose_,
              false,
          },
          values_{fieldName + "_values", db,       exec_,
                  global_matrix_index_,  verbose_, true},
          col_idxs_{fieldName + "_cols",  db,       exec_,
                    global_matrix_index_, verbose_, false},
          row_idxs_{fieldName + "_rows",  db,       exec_,
                    global_matrix_index_, verbose_, false}
    {
        std::cout << "!!!!! DEBUG nElems_ + interface "
                  << nElems_ + nInterfaces_ << "\n";
        if (!col_idxs_.get_stored() || col_idxs_.get_update()) {
            auto other_proc_cell_ids =
                get_other_proc_cell_ids(nInterfaces_, interfaces);
            SIMPLE_TIME(
                verbose_, init_host_sparsity_pattern,
                init_host_sparsity_pattern(interfaces, other_proc_cell_ids);)
        }
        if (!values_.get_stored() || values_.get_update()) {
            auto other_proc_bou_coeffs = get_other_proc_bou_coeffs(
                nInterfaces_, interfaces, interfaceBouCoeffs);
            SIMPLE_TIME(
                verbose_, update_host_matrix_data,
                update_host_matrix_data(interfaces, other_proc_bou_coeffs);)
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
          global_interface_index_{nInterfaces_},
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          ldu_csr_idx_mapping_{fieldName + "ldu_csr_map", db,       exec_,
                               global_matrix_index_,      verbose_, false},
          ldu_csr_idx_interface_mapping_{
              fieldName + "_ldu_csr_interface_map",
              db,
              exec_,
              global_interface_index_,
              verbose_,
              false,
          },
          values_{fieldName + "values", db,       exec_,
                  global_matrix_index_, verbose_, true},
          col_idxs_{fieldName + "cols",   db,       exec_,
                    global_matrix_index_, verbose_, false},
          row_idxs_{fieldName + "rows",   db,       exec_,
                    global_matrix_index_, verbose_, false}
    {
        // TODO reimplement this
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

    // TODO merge with get_other_proc_cell_ids
    std::vector<scalar> get_other_proc_bou_coeffs(
        const label nInterfaces, const lduInterfaceFieldPtrsList &interfaces,
        const FieldField<Field, scalar> interfaceBouCoeffs)
    {
        std::vector<scalar> ret{};
        ret.reserve(nInterfaces);

        label startOfRequests = Pstream::nRequests();
        for (int i = 0; i < interfaces.size(); i++) {
            if (interfaces.operator()(i) == nullptr) {
                continue;
            }

            const auto iface{interfaces.operator()(i)};
            const auto &face_cells{interfaceBouCoeffs[i]};
            const label interface_size = face_cells.size();

            if (isA<processorLduInterface>(iface->interface())) {
                const processorLduInterface &pldui =
                    refCast<const processorLduInterface>(iface->interface());
                const label neighbProcNo = pldui.neighbProcNo();

                word msg = "send face cells interface " + std::to_string(i) +
                           " from proc " + std::to_string(Pstream::myProcNo()) +
                           " to neighbour proc " + std::to_string(neighbProcNo);

                LOG_2(verbose_, msg)
                pldui.send(Pstream::commsTypes::nonBlocking, face_cells);
            }
        }
        Pstream::waitRequests(startOfRequests);
        LOG_2(verbose_, "send face cells done")

        for (int i = 0; i < interfaces.size(); i++) {
            if (interfaces.operator()(i) == nullptr) {
                continue;
            }

            const auto iface{interfaces.operator()(i)};
            const auto &face_cells{iface->interface().faceCells()};
            const label interface_size = face_cells.size();

            if (isA<processorLduInterface>(iface->interface())) {
                const processorLduInterface &pldui =
                    refCast<const processorLduInterface>(iface->interface());
                const label neighbProcNo = pldui.neighbProcNo();

                word msg_2 = "receive face cells interface " +
                             std::to_string(i) + " from proc " +
                             std::to_string(neighbProcNo);
                LOG_2(verbose_, msg_2)

                auto otherSide_tmp = pldui.receive<scalar>(
                    Pstream::commsTypes::nonBlocking, interface_size);
                LOG_2(verbose_, "receive face cells done")

                for (label cellI = 0; cellI < interface_size; cellI++) {
                    ret.push_back(otherSide_tmp()[cellI]);
                }
            }
        }
        word msg = "done collecting neighbouring processor cell id";

        LOG_2(verbose_, msg)
        return ret;
    }

    std::vector<label> get_other_proc_cell_ids(
        const label nInterfaces, const lduInterfaceFieldPtrsList &interfaces)
    {
        std::vector<label> ret{};
        ret.reserve(nInterfaces);

        label startOfRequests = Pstream::nRequests();
        for (int i = 0; i < interfaces.size(); i++) {
            if (interfaces.operator()(i) == nullptr) {
                continue;
            }

            const auto iface{interfaces.operator()(i)};
            const auto &face_cells{iface->interface().faceCells()};
            const label interface_size = face_cells.size();

            if (isA<processorLduInterface>(iface->interface())) {
                const processorLduInterface &pldui =
                    refCast<const processorLduInterface>(iface->interface());
                const label neighbProcNo = pldui.neighbProcNo();

                word msg = "send face cells interface " + std::to_string(i) +
                           " from proc " + std::to_string(Pstream::myProcNo()) +
                           " to neighbour proc " + std::to_string(neighbProcNo);

                LOG_2(verbose_, msg)
                pldui.send(Pstream::commsTypes::nonBlocking, face_cells);
            }
        }
        Pstream::waitRequests(startOfRequests);
        LOG_2(verbose_, "send face cells done")

        for (int i = 0; i < interfaces.size(); i++) {
            if (interfaces.operator()(i) == nullptr) {
                continue;
            }

            const auto iface{interfaces.operator()(i)};
            const auto &face_cells{iface->interface().faceCells()};
            const label interface_size = face_cells.size();

            if (isA<processorLduInterface>(iface->interface())) {
                const processorLduInterface &pldui =
                    refCast<const processorLduInterface>(iface->interface());
                const label neighbProcNo = pldui.neighbProcNo();

                word msg_2 = "receive face cells interface " +
                             std::to_string(i) + " from proc " +
                             std::to_string(neighbProcNo);
                LOG_2(verbose_, msg_2)

                auto otherSide_tmp = pldui.receive<label>(
                    Pstream::commsTypes::nonBlocking, interface_size);
                LOG_2(verbose_, "receive face cells done")

                for (label cellI = 0; cellI < interface_size; cellI++) {
                    ret.push_back(otherSide_tmp()[cellI]);
                }
            }
        }
        word msg = "done collecting neighbouring processor cell id";

        LOG_2(verbose_, msg)
        return ret;
    }

    // TODO this is pretty much the same as get_other_proc_cell_ids
    // and could probably trimmed down a lot
    void insert_interface_coeffs(const lduInterfaceFieldPtrsList &interfaces,
                                 const std::vector<label> &other_proc_cell_ids,
                                 int *rows, int *cols, label row,
                                 label &element_ctr,
                                 label *sorting_interface_idxs,
                                 const bool upper) const
    {
        label interface_ctr = 0;
        for (int i = 0; i < interfaces.size(); i++) {
            if (interfaces.operator()(i) == nullptr) {
                continue;
            }

            const auto &iface{interfaces.operator()(i)};
            const auto &face_cells{iface->interface().faceCells()};
            const label interface_size = face_cells.size();

            if (isA<processorLduInterface>(iface->interface())) {
                const processorLduInterface &pldui =
                    refCast<const processorLduInterface>(iface->interface());

                // if rank of corresponding processor is greater
                // then own processor idx are not on lower matrix row
                const label neighbProcNo = pldui.neighbProcNo();
                if (upper) {
                    if (neighbProcNo > Pstream::myProcNo()) {
                        interface_ctr += interface_size;
                        continue;
                    }

                } else {
                    if (neighbProcNo < Pstream::myProcNo()) {
                        interface_ctr += interface_size;
                        continue;
                    }
                }

                // check if current cell is on the current patch
                // NOTE cells can be several times on same patch
                for (label cellI = 0; cellI < interface_size; cellI++) {
                    if (face_cells[cellI] == row) {
                        const label other_side_global_cellID =
                            global_cell_index_.toGlobal(
                                neighbProcNo,
                                other_proc_cell_ids[interface_ctr + cellI]);
                        std::cout << " Pstream::myProcNo() "
                                  << Pstream::myProcNo() << " row " << row
                                  << " face_cells[cellI]" << face_cells[cellI]
                                  << " other_proc_cell_ids[interface_ctr + "
                                     "cellI] "
                                  << other_proc_cell_ids[interface_ctr + cellI]
                                  << " other_side_global_cellID "
                                  << other_side_global_cellID << std::endl;

                        rows[element_ctr] = global_cell_index_.toGlobal(row);
                        cols[element_ctr] = other_side_global_cellID;

                        sorting_interface_idxs[interface_ctr + cellI] =
                            element_ctr;

                        element_ctr++;
                    }
                }
                interface_ctr += interface_size;
            }
        }
    }

    void init_host_sparsity_pattern(
        const lduInterfaceFieldPtrsList &interfaces,
        const std::vector<label> other_proc_cell_ids) const
    {
        // Step 1 local ldu -> csr conversion
        // including local processor offset
        LOG_1(verbose_, "start init host sparsity pattern")

        auto lower_local = idx_array::view(
            exec_.get_ref_exec(), nNeighbours_,
            const_cast<label *>(&this->matrix().lduAddr().lowerAddr()[0]));

        auto upper_local = idx_array::view(
            exec_.get_ref_exec(), nNeighbours_,
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
        auto sorting_interface_idxs = ldu_csr_idx_interface_mapping_.get_data();

        for (label row = 0; row < nCells_; row++) {
            // check for lower idxs
            insert_interface_coeffs(interfaces, other_proc_cell_ids, rows, cols,
                                    row, element_ctr, sorting_interface_idxs,
                                    false);

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

            insert_interface_coeffs(interfaces, other_proc_cell_ids, rows, cols,
                                    row, element_ctr, sorting_interface_idxs,
                                    true);
        }
        LOG_1(verbose_, "done init host matrix")
    }

    bool get_verbose() const { return verbose_; }

    void update_host_matrix_data(
        const lduInterfaceFieldPtrsList &interfaces,
        const std::vector<scalar> &interfaceBouCoeffs) const
    {
        auto ref_exec = gko::ReferenceExecutor::create();
        // TODO create in ctr
        // as devicePersistent so that we can reuse the memory

        auto values = values_.get_data();

        const auto sorting_idxs = ldu_csr_idx_mapping_.get_const_data();
        const auto sorting_interface_idxs =
            ldu_csr_idx_interface_mapping_.get_const_data();

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

        label interface_ctr{0};
        for (int i = 0; i < interfaces.size(); i++) {
            if (interfaces.operator()(i) == nullptr) {
                continue;
            }
            const auto iface{interfaces.operator()(i)};
            const label patch_size = iface->interface().faceCells().size();

            if (!isA<processorLduInterface>(iface->interface())) {
                continue;
            }

            for (label cellI = 0; cellI < patch_size; cellI++) {
                values[sorting_interface_idxs[interface_ctr + cellI]] =
                    -interfaceBouCoeffs[interface_ctr + cellI];
            }
            interface_ctr += patch_size;
        }
    }

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
