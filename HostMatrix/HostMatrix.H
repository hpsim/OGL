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
        const lduInterfaceFieldPtrsList &interfaces_) const;

    // TODO merge with get_other_proc_cell_ids
    std::vector<scalar> get_other_proc_bou_coeffs(
        const label nInterfaces, const lduInterfaceFieldPtrsList &interfaces,
        const FieldField<Field, scalar> interfaceBouCoeffs);

    std::vector<label> get_other_proc_cell_ids(
        const label nInterfaces, const lduInterfaceFieldPtrsList &interfaces);

    // TODO this is pretty much the same as get_other_proc_cell_ids
    // and could probably trimmed down a lot
    void insert_interface_coeffs(const lduInterfaceFieldPtrsList &interfaces,
                                 const std::vector<label> &other_proc_cell_ids,
                                 int *rows, int *cols, label row,
                                 label &element_ctr,
                                 label *sorting_interface_idxs,
                                 const bool upper) const;

    void init_host_sparsity_pattern(
        const lduInterfaceFieldPtrsList &interfaces,
        const std::vector<label> other_proc_cell_ids) const;

    bool get_verbose() const { return verbose_; }

    void update_host_matrix_data(
        const lduInterfaceFieldPtrsList &interfaces,
        const std::vector<scalar> &interfaceBouCoeffs) const;

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
