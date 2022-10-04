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


struct PersistentSparsityPattern {
    PersistentSparsityPattern(
      const word &fieldName,
      const objectRegistry &db,
      const ExecutorHandler &exec,
      const label size,
      const label verbose):
    col_idxs_{
        fieldName + "_cols",
        db,
        exec,
        size,
        verbose,
        false,  // For now we assume columns and rows to be constant
        false   // same as for values, leave on device
    },
    row_idxs_{
        fieldName + "_rows",
        db,
        exec,
        size,
        verbose,
        false,  // For now we assume columns and rows to be constant
        false   // same as for values, leave on device
  }
    {}

    mutable PersistentArray<label> col_idxs_;

    mutable PersistentArray<label> row_idxs_;
};

/* The HostMatrixWrapper class handles the conversion from OpenFOAMs lduMatrix
 * format into Ginkgo array data structures
 *
 * It mainly generates:
 * - sparsity pattern/device_matrix_data of the local_matrix and
 * non_local_matrix (with global columns)
 * - send_sizes
 * - recv_sizes
 *
 * */
template <class MatrixType>
class HostMatrixWrapper : public MatrixType::solver {
private:
    using vec = gko::matrix::Dense<scalar>;
    // dimensions of the local host matrix
    //
    const ExecutorHandler exec_;

    const label verbose_;

    //  local matrix indices
    const label nrows_;

    const label upper_nnz_;

    const label non_diag_nnz_;

    const label nnz_local_matrix_;

    const gkoGlobalIndex global_row_index_;

    mutable PersistentSparsityPattern local_sparsity_;

    mutable PersistentArray<scalar> local_coeffs_;

    mutable PersistentArray<label> local_ldu_csr_idx_mapping_;
    // non-local indices
    const label nnz_non_local_matrix_;

    mutable PersistentSparsityPattern non_local_sparsity_;

    mutable PersistentArray<scalar> non_local_coeffs_;

    const word permutation_matrix_name_;

    const bool permutation_stored_;

    mutable std::shared_ptr<gko::LinOp> P_;


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
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          nrows_(matrix.diag().size()),
          upper_nnz_(matrix.lduAddr().upperAddr().size()),
          non_diag_nnz_(2 * upper_nnz_),
          nnz_local_matrix_(nrows_ + 2 * upper_nnz_),
          global_row_index_{nrows_},
          local_sparsity_{
              fieldName + "_local", db, exec_, nnz_local_matrix_, verbose_,
          },
          local_coeffs_{
              fieldName + "_local_coeffs",
              db,
              exec_,
              nnz_local_matrix_,
              verbose_,
              true,  // needs to be updated
              false  // leave it on host once it is turned into a distributed
                     // matrix it will be put on the device
          },
          local_ldu_csr_idx_mapping_{
              fieldName + "_ldu_csr_map",
              db,
              exec_,
              nnz_local_matrix_,
              verbose_,
              false,
              false,
          },
          nnz_non_local_matrix_(compute_non_local_nnz(interfaces)),
          non_local_sparsity_{
              fieldName + "_non_local", db,       exec_,
              nnz_non_local_matrix_,    verbose_,
          },
          non_local_coeffs_{
              fieldName + "_non_local_coeffs",
              db,
              exec_,
              nnz_non_local_matrix_,
              verbose_,
              true,  // needs to be updated
              false  // leave it on host once it is turned into a distributed
                     // matrix it will be put on the device
          },
          permutation_matrix_name_{"PermutationMatrix"},
          permutation_stored_{
              db.template foundObject<regIOobject>(permutation_matrix_name_)},
          P_{(permutation_stored_) ? db.template lookupObjectRef<
                                           DevicePersistentBase<gko::LinOp>>(
                                           permutation_matrix_name_)
                                         .get_ptr()
                                   : nullptr}
    {
        if (!local_sparsity_.col_idxs_.get_stored() ||
            local_sparsity_.col_idxs_.get_update()) {
            SIMPLE_TIME(verbose_, init_local_sparsity_pattern,
                        init_local_sparsity_pattern();)
            SIMPLE_TIME(verbose_, init_non_local_sparsity_pattern,
                        init_non_local_sparsity_pattern(interfaces);)
        }
        if (!local_coeffs_.get_stored() || local_coeffs_.get_update()) {
            SIMPLE_TIME(verbose_, update_local_matrix_data,
                        update_local_matrix_data();)
            SIMPLE_TIME(
                verbose_, update_non_local_matrix_data,
                update_non_local_matrix_data(interfaces, interfaceBouCoeffs);)
        }
    }

    // TODO reimplement this at some point
    // coupled  wrapper constructor
    HostMatrixWrapper(const objectRegistry &db, const MatrixType &matrix,
                      const dictionary &solverControls, const word &fieldName)
        : MatrixType::solver(fieldName, matrix, solverControls),
          exec_{db, solverControls, fieldName},
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          nrows_(matrix.diag().size()),
          upper_nnz_(matrix.lduAddr().upperAddr().size()),
          non_diag_nnz_(2 * upper_nnz_),
          nnz_local_matrix_(nrows_ + 2 * upper_nnz_),
          global_row_index_{nrows_},
          local_sparsity_{
              fieldName + "_cols", db, exec_, nnz_local_matrix_, verbose_,
          },
          local_coeffs_{
              fieldName + "_coeffs",
              db,
              exec_,
              nnz_local_matrix_,
              verbose_,
              true,  // needs to be updated
              false  // leave it on host once it is turned into a distributed
                     // matrix it will be put on the device
          },
          local_ldu_csr_idx_mapping_{
              fieldName + "_ldu_csr_map",
              db,
              exec_,
              nnz_local_matrix_,
              verbose_,
              false,
              false,
          },
          nnz_non_local_matrix_(),
          non_local_sparsity_{
              fieldName + "_non_local", db,       exec_,
              nnz_non_local_matrix_,    verbose_,
          },
          non_local_coeffs_{
              fieldName + "_non_local_coeffs",
              db,
              exec_,
              nnz_non_local_matrix_,
              verbose_,
              true,  // needs to be updated
              false  // leave it on host once it is turned into a distributed
                     // matrix it will be put on the device
          },
          permutation_matrix_name_{"PermutationMatrix"},
          permutation_stored_{
              db.template foundObject<regIOobject>(permutation_matrix_name_)},
          P_{(permutation_stored_) ? db.template lookupObjectRef<
                                           DevicePersistentBase<gko::LinOp>>(
                                           permutation_matrix_name_)
                                         .get_ptr()
                                   : nullptr}
    {
        // TODO reimplement this
    }


    /* Iterates all interfaces and counts elements
    **
    */
    label compute_non_local_nnz(
        const lduInterfaceFieldPtrsList &interfaces_) const;

    /* Collect the non local matrix coefficients
    **
    */
    std::vector<scalar> communicate_non_local_coeffs(
        const lduInterfaceFieldPtrsList &interfaces,
        const FieldField<Field, scalar> &interfaceBouCoeffs) const;

    /* Iterates all interfaces and collects non-local global column indices
    **
    ** Non-local global column indices are computed from the global cell-id
    ** (row) on the other side of an interface
    **
    ** @return vector off size nnz_non_local, with sections for each interface
    ** ret = [(1,2,3),(10,20,30), ...]
    **         i0      i1,       i...
    */
    std::vector<std::tuple<label, label>> communicate_non_local_col_indices(
        const lduInterfaceFieldPtrsList &interfaces) const;

    /* Based on OpenFOAMs ldu matrix format this function computes two
     *consecutive index arrays in row major ordering and scattering indices
     **
     */
    void init_local_sparsity_pattern() const;

    void init_non_local_sparsity_pattern(
        const lduInterfaceFieldPtrsList &interfaces) const;

    // // TODO this is pretty much the same as
    // communicate_non_local_row_indices
    // // and could probably trimmed down a lot
    // void insert_interface_coeffs(const lduInterfaceFieldPtrsList
    // &interfaces,
    //                              const std::vector<label>
    //                              &other_proc_cell_ids, int *rows, int
    //                              *cols, label row, label global_row,
    //                              label &element_ctr, label
    //                              *sorting_interface_idxs, const bool
    //                              upper) const;


    bool get_verbose() const { return verbose_; }

    // updater

    void update_local_matrix_data() const;

    void update_non_local_matrix_data(
        const lduInterfaceFieldPtrsList &interfaces,
        const FieldField<Field, scalar> &interfaceBouCoeffs) const;

    // getter

    label get_local_nrows() const { return nrows_; }

    label get_global_nrows() const { return global_row_index_.size(); }

    const ExecutorHandler &get_exec_handler() const { return exec_; }

    const gkoGlobalIndex &get_global_row_index() const
    {
        return global_row_index_;
    }


    PersistentArray<scalar> &get_local_coeffs() const  //
    {
        return local_coeffs_;
    };

    PersistentArray<label> &get_local_col_idxs() const
    {
        return local_sparsity_.col_idxs_;
    };

    PersistentArray<label> &get_local_row_idxs() const
    {
        return local_sparsity_.row_idxs_;
    };

    PersistentArray<scalar> &get_non_local_coeffs() const
    {
        return non_local_coeffs_;
    };

    PersistentArray<label> &get_non_local_col_idxs() const
    {
        return non_local_sparsity_.col_idxs_;
    };

    PersistentArray<label> &get_non_local_row_idxs() const
    {
        return non_local_sparsity_.row_idxs_;
    };
};


}  // namespace Foam
#endif
