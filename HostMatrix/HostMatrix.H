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
#include "../PersistentDeviceIdGuard/DeviceIdGuard.H"
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
    },
    ldu_mapping_{
        fieldName + "_ldu_map",
        db,
        exec,
        size,
        verbose,
        false,
        false
    }
    {}

    mutable PersistentArray<label> col_idxs_;

    mutable PersistentArray<label> row_idxs_;

    // ldu_mapping[sorted_(csr)_position] =
    // unsorted_(consecutive_ldu_)_position
    mutable PersistentArray<label> ldu_mapping_;
};

/* The HostMatrixWrapper class handles the conversion from OpenFOAMs lduMatrix
 * format into Ginkgo array data structures
 *
 * It mainly generates:
 * - sparsity pattern/device_matrix_data of the local_matrix and
 * non_local_matrix (with global columns)
 * */
template <class MatrixType>
class HostMatrixWrapper : public MatrixType::solver {
private:
    using vec = gko::matrix::Dense<scalar>;

    const ExecutorHandler exec_;

    const DeviceIdGuardHandler device_id_guard_;

    const label verbose_;

    // multiply the complete system by this factor, ie sAx=sb
    // NOTE this could be also achieved by just fliping the sign
    const scalar scaling_;

    // number of local matrix rows
    const label nrows_;

    // number of local elements on interfaces
    // ie number of interfaces which column_idx is < nrows_
    // and have to be sorted into local matrix
    const label local_interface_nnz_;

    // number of local upper elements
    // ie coefficients which column_idx < nrows_
    const label upper_nnz_;

    // total number of local upper and lower elements
    // ie 2*upper_nnz_ since the sparsity pattern is symmetric
    // TODO should this include the local_interface_nnz_ ?
    const label non_diag_nnz_;

    // nnz of local matrix wo local interfaces
    const label local_matrix_nnz_;

    // nnzs of local matrix including local interfaces
    const label local_matrix_w_interfaces_nnz_;

    const gkoGlobalIndex global_row_index_;

    mutable PersistentSparsityPattern local_sparsity_;

    mutable PersistentArray<scalar> local_coeffs_;

    // non-local indices
    const label non_local_matrix_nnz_;

    mutable PersistentSparsityPattern non_local_sparsity_;

    mutable PersistentArray<scalar> non_local_coeffs_;

    const word permutation_matrix_name_;

    const bool permutation_stored_;

    mutable std::shared_ptr<gko::LinOp> P_;


public:
    // segregated wrapper constructor
    // TODO - compute local_interface_nnzs these are matrix coefficient
    //      that are on an interface but not on a different rank
    //      - then a new array/vector is needed to store local_interfaces
    //      - if local interfaces are present init_local_sparsity_pattern
    //      needs to be adapted
    //      - local_sparsity_.ldu_mapping needs a new section for interfaces
    HostMatrixWrapper(
        const objectRegistry &db, const MatrixType &matrix,
        // coeffs for cells on boundaries
        const FieldField<Field, scalar> &interfaceBouCoeffs,
        // coeffs for internal cells
        const FieldField<Field, scalar> &interfaceIntCoeffs,
        // pointers to interfaces can be used to access concrete
        // functions such as transferring indices, patch neighbours etc
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls, const word &fieldName)
        : MatrixType::solver(fieldName, matrix, interfaceBouCoeffs,
                             interfaceIntCoeffs, interfaces, solverControls),
          exec_{db, solverControls, fieldName},
          device_id_guard_{db, fieldName, exec_.get_device_exec()},
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          scaling_(solverControls.lookupOrDefault<scalar>("scaling", 1)),
          nrows_(matrix.diag().size()),
          local_interface_nnz_(count_interface_nnz(interfaces, false)),
          upper_nnz_(matrix.lduAddr().upperAddr().size()),
          non_diag_nnz_(2 * upper_nnz_),
          local_matrix_nnz_(nrows_ + 2 * upper_nnz_),
          local_matrix_w_interfaces_nnz_(local_matrix_nnz_ +
                                         local_interface_nnz_),
          global_row_index_{nrows_},
          local_sparsity_{
              fieldName + "_local",           db,       exec_,
              local_matrix_w_interfaces_nnz_, verbose_,
          },
          local_coeffs_{
              fieldName + "_local_coeffs",
              db,
              exec_,
              local_matrix_w_interfaces_nnz_,
              verbose_,
              true,  // needs to be updated
              false  // leave it on host once it is turned into a distributed
                     // matrix it will be put on the device
          },
          non_local_matrix_nnz_(count_interface_nnz(interfaces, true)),
          non_local_sparsity_{
              fieldName + "_non_local", db,       exec_,
              non_local_matrix_nnz_,    verbose_,
          },
          non_local_coeffs_{
              fieldName + "_non_local_coeffs",
              db,
              exec_,
              non_local_matrix_nnz_,
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
            TIME_WITH_FIELDNAME(verbose_, init_local_sparsity_pattern,
                                this->fieldName(),
                                init_local_sparsity_pattern(interfaces);)
            TIME_WITH_FIELDNAME(verbose_, init_non_local_sparsity_pattern,
                                this->fieldName(),
                                init_non_local_sparsity_pattern(interfaces);)
        }
        if (!local_coeffs_.get_stored() || local_coeffs_.get_update()) {
            TIME_WITH_FIELDNAME(
                verbose_, update_local_matrix_data, this->fieldName(),
                update_local_matrix_data(interfaces, interfaceBouCoeffs);)
            TIME_WITH_FIELDNAME(
                verbose_, update_non_local_matrix_data, this->fieldName(),
                update_non_local_matrix_data(interfaces, interfaceBouCoeffs);)
        }
    }

    // TODO reimplement this at some point
    // coupled  wrapper constructor
    HostMatrixWrapper(const objectRegistry &db, const MatrixType &matrix,
                      const dictionary &solverControls, const word &fieldName)
        : MatrixType::solver(fieldName, matrix, solverControls),
          exec_{db, solverControls, fieldName},
          device_id_guard_{db, fieldName, exec_.get_device_exec()},
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          scaling_(solverControls.lookupOrDefault<scalar>("scaling", 1)),
          nrows_(matrix.diag().size()),
          local_interface_nnz_(0),
          upper_nnz_(matrix.lduAddr().upperAddr().size()),
          non_diag_nnz_(2 * upper_nnz_),
          local_matrix_nnz_(nrows_ + 2 * upper_nnz_),
          local_matrix_w_interfaces_nnz_(local_matrix_nnz_ +
                                         local_interface_nnz_),
          global_row_index_{nrows_},
          local_sparsity_{
              fieldName + "_cols", db, exec_, local_matrix_nnz_, verbose_,
          },
          local_coeffs_{
              fieldName + "_coeffs",
              db,
              exec_,
              local_matrix_nnz_,
              verbose_,
              true,  // needs to be updated
              false  // leave it on host once it is turned into a distributed
                     // matrix it will be put on the device
          },
          non_local_matrix_nnz_(),
          non_local_sparsity_{
              fieldName + "_non_local", db,       exec_,
              non_local_matrix_nnz_,    verbose_,
          },
          non_local_coeffs_{
              fieldName + "_non_local_coeffs",
              db,
              exec_,
              non_local_matrix_nnz_,
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

    /* Iterates all interfaces and collects the coefficients into a vector
    **
    ** @param local whether local or non local coefficients should be collected
    */
    std::vector<scalar> collect_interface_coeffs(
        const lduInterfaceFieldPtrsList &interfaces_,
        const FieldField<Field, scalar> &interfaceBouCoeffs,
        const bool local) const;


    /* Iterates all local interfaces and returns the relative order and
     *corresponding row and column indices
     **
     */
    std::vector<std::tuple<label, label, label>>
    collect_local_interface_indices(
        const lduInterfaceFieldPtrsList &interfaces_) const;


    /* Iterates all interfaces and counts the number of elements
    **
    ** @param interfaces The list of interfaces for the search
    ** @param proc_interfaces Count only elements on (true)
    *processorLduInterfaces or exclude processorLduInterfaces (false)
    */
    label count_interface_nnz(const lduInterfaceFieldPtrsList &interfaces,
                              bool proc_interfaces) const;

    /* Iterates all interfaces and collects non-local global column indices
    **
    ** Non-local global column indices are computed from the global cell-id
    ** (row) on the other side of an interface
    **
    ** @return vector off size nnz_non_local, with sections for each interface
    ** ret = [(1,2,3),(10,20,30), ...]
    **         i0      i1,       i...
    */
    std::vector<std::tuple<label, label, label>> collect_non_local_col_indices(
        const lduInterfaceFieldPtrsList &interfaces) const;

    /* Based on OpenFOAMs ldu matrix format this function computes two
     *consecutive index arrays in row major ordering and scattering indices
     **
     */
    void init_local_sparsity_pattern(
        const lduInterfaceFieldPtrsList &interfaces) const;

    void init_non_local_sparsity_pattern(
        const lduInterfaceFieldPtrsList &interfaces) const;


    bool get_verbose() const { return verbose_; }

    // updater

    void update_local_matrix_data(
        const lduInterfaceFieldPtrsList &interfaces,
        const FieldField<Field, scalar> &interfaceBouCoeffs) const;

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
