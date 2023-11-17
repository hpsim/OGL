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
#include "DevicePersistent/Array/Array.H"
#include "DevicePersistent/DeviceIdGuard/DeviceIdGuard.H"
#include "DevicePersistent/IOGlobalIndex/gkoGlobalIndex.H"
#include "common/common.H"


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

struct CommunicationPattern {
    // an array storing to which rank to communicate
    gko::array<label> target_ids;

    // an array storing how many elements to communicate
    // to the corresponding target_id
    gko::array<label> target_sizes;

    // send_idx stores the index_set ie which cells
    // are owned by the interface and the corresponding target rank
    std::vector<std::pair<gko::array<label>, label>> send_idxs;
};


// Free functions
const lduInterfaceField *interface_getter(
    const lduInterfaceFieldPtrsList &interfaces, const label i);

/** Write contiguous row and col indices from OpenFOAM lower and upper indices
 **
 **
 ** @param nrows number of rows
 ** @param upper_nnz number of non zeros in the upper triangular matrix
 ** @param is_symmetric whether matrix is symmetric, in the symmetric case
 ** the lower elements indices in the permute array are computed differently
 ** @param upper pointer to array of
 ** @param lower pointer to rows array
 ** @param rows pointer to rows array
 ** @param cols pointer to columns array
 ** @param permute pointer to permuter array
 */
void init_local_sparsity(const label nrows, const label upper_nnz,
                         const bool is_symmetric, const label *upper,
                         const label *lower, label *rows, label *cols,
                         label *permute);

/** Perform update of the coefficients array in row major order from a symmetric
 *ldu matrix
 **
 ** @note this assumes out to be on the host
 ** @note performs contiguous writes into out and random access of diag and
 *upper
 ** @param total_nnz number of non zeros entries in the matrix
 ** @param upper_nnz number of non zeros entries in the upper triangular matrix
 ** @param permute mapping between the original position in the ldu input data
 **        and the position in the row major ordered output data
 **        ie. permute[i] maps out[i] to original position in diag or upper
 **        with
 **        upper_nnz <= permute < upper_nnz + diag_nnz for diag entries
 ** @param scale scaling factor to scale the matrix entries
 ** @param diag pointer to diagonal entries of the original ldu matrix
 ** @param upper pointer to upper triangular entries of the original ldu matrix
 ** @param out pointer to data of output matrix
 **/
void symmetric_update(const label total_nnz, const label upper_nnz,
                      const label *permute, const scalar scale,
                      const scalar *diag, const scalar *upper, scalar *out);


/** Perform update of the coefficients array in row major order from a
 *non-symmetric ldu matrix
 **
 ** @note this assumes out to be on the host
 ** @note performs contiguous writes into out and random access of diag and
 *upper
 ** @param total_nnz number of non zeros entries in the matrix
 ** @param upper_nnz number of non zeros entries in the upper triangular matrix
 ** @param permute mapping between the original position in the ldu input data
 **        and the position in the row major ordered output data
 **        ie. permute[i] maps out[i] to original position in diag or upper
 **        with
 **        upper_nnz <= permute < upper_nnz + diag_nnz for diag entries
 **        upper_nnz + diag_nnz <= permute for lower entries
 ** @param scale scaling factor to scale the matrix entries
 ** @param diag pointer to diagonal entries of the original ldu matrix
 ** @param upper pointer to upper triangular entries of the original ldu matrix
 ** @param lower pointer to lower triangular entries of the original ldu matrix
 ** @param out pointer to data of output matrix
 **/
void non_symmetric_update(const label total_nnz, const label upper_nnz,
                          const label *permute, const scalar scale,
                          const scalar *diag, const scalar *upper,
                          const scalar *lower, scalar *dense);


/** Perform update of the coefficients array in row major order from a
 ** symmetric ldu matrix including local interface data.
 **
 ** @note this assumes out to be on the host
 ** @note performs contiguous writes into out and random access of diag and
 ** upper
 **
 ** @param total_nnz number of non zeros entries in the matrix
 ** @param upper_nnz number of non zeros entries in the upper triangular matrix
 ** @param permute mapping between the original position in the ldu input data
 **        and the position in the row major ordered output data
 **        ie. permute[i] maps out[i] to original position in diag or upper
 **        with
 **        upper_nnz <= permute < upper_nnz + diag_nnz for diag entries
 **        upper_nnz + diag_nnz <= permute for interface entries
 ** @param scale scaling factor to scale the matrix entries
 ** @param diag pointer to diagonal entries of the original ldu matrix
 ** @param upper pointer to upper triangular entries of the original ldu matrix
 ** @param interface pointer to interface entries
 ** @param out pointer to data of output matrix
 **/
void symmetric_update_w_interface(const label total_nnz, const label diag_nnz,
                                  const label upper_nnz, const label *permute,
                                  const scalar scale, const scalar *diag,
                                  const scalar *upper, const scalar *interface,
                                  scalar *dense);


/** Perform update of the coefficients array in row major order from a
 ** non-symmetric ldu matrix including local interface data.
 **
 ** @note this assumes out to be on the host
 ** @note performs contiguous writes into out and random access of diag and
 ** upper
 ** @param total_nnz number of non zeros entries in the matrix
 ** @param upper_nnz number of non zeros entries in the upper triangular matrix
 ** @param permute mapping between the original position in the ldu input data
 **        and the position in the row major ordered output data
 **        ie. permute[i] maps out[i] to original position in diag or upper
 **        with
 **        upper_nnz <= permute < upper_nnz + diag_nnz for diag entries
 **        upper_nnz + diag_nnz <= permute for lower entries
 **        2*upper_nnz + diag_nnz <= permute for interface entries
 ** @param scale scaling factor to scale the matrix entries
 ** @param diag pointer to diagonal entries of the original ldu matrix
 ** @param upper pointer to upper triangular entries of the original ldu matrix
 ** @param lower pointer to lower triangular entries of the original ldu matrix
 ** @param interface pointer to interface entries
 ** @param out pointer to data of output matrix
 **/
void non_symmetric_update_w_interface(
    const label total_nnz, const label diag_nnz, const label upper_nnz,
    const label *permute, const scalar scale, const scalar *diag,
    const scalar *upper, const scalar *lower, const scalar *couple_coeffs,
    scalar *dense);


/* The HostMatrixWrapper class handles the conversion from OpenFOAMs lduMatrix
 * format into Ginkgo array data structures
 *
 * It mainly generates:
 * - sparsity pattern/device_matrix_data of the local_matrix and
 * - non_local_matrix (with global columns)
 * - CommunicationPattern
 * */
template <class MatrixType>
class HostMatrixWrapper : public MatrixType::solver {
private:
    using vec = gko::matrix::Dense<scalar>;

    const ExecutorHandler exec_;

    const DeviceIdGuardHandler device_id_guard_;

    const label verbose_;

    // Whether the matrix coefficients should be reordered
    // during copy or on device
    // NOTE Not implemented
    const bool reorder_on_copy_;

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

    mutable PersistentSparsityPattern local_sparsity_;

    mutable PersistentArray<scalar> local_coeffs_;

    // non-local indices
    const label non_local_matrix_nnz_;

    const CommunicationPattern communication_pattern_;

    mutable PersistentSparsityPattern non_local_sparsity_;

    mutable PersistentArray<scalar> non_local_coeffs_;

    // FIXME TODO unused
    const word permutation_matrix_name_;

    const bool permutation_stored_;

    mutable std::shared_ptr<gko::LinOp> P_;


    /* Iterates all interfaces and collects the coefficients into a vector
    **
    ** @param local whether local or non local coefficients should be collected
    */
    std::vector<scalar> collect_interface_coeffs(
        const lduInterfaceFieldPtrsList &interfaces_,
        const FieldField<Field, scalar> &interfaceBouCoeffs,
        const bool local) const;


    /* Iterates all interfaces and counts the number of elements
    **
    ** @param interfaces The list of interfaces for the search
    ** @param proc_interfaces Count only elements on (true)
    *processorLduInterfaces or exclude processorLduInterfaces (false)
    */
    label count_interface_nnz(const lduInterfaceFieldPtrsList &interfaces,
                              bool proc_interfaces) const;

    /** Iterates all local interfaces and returns the relative order and
    **corresponding row and column indices
    **
    ** @return vector of tuples containing the interface number, the local row,
    *the local column
    **/
    std::vector<std::tuple<label, label, label>>
    collect_local_interface_indices(
        const lduInterfaceFieldPtrsList &interfaces_) const;


    /** Iterates all interfaces and counts the number of unique neighbour
     **processors and number of interfaces in total for this processor
     ** and collects all interface cells of this rank
     **
     ** @param interfaces The list of interfaces for the search
     ** @return the CommunicationPattern
     */
    CommunicationPattern create_communication_pattern(
        const lduInterfaceFieldPtrsList &interfaces) const;

    /** Iterates all interfaces and collect the corresponding cell id (row)
    ** and a unique counter
    **
    ** @return vector off size nnz_non_local, with sections for each interface
    ** ret = [(1,2),(2,20), ...]
    **         i0   i1,   i...
    */
    std::vector<std::tuple<label, label>> collect_cells_on_interface(
        const lduInterfaceFieldPtrsList &interfaces) const;

    /** Based on OpenFOAMs ldu matrix format this function computes two
     ** consecutive index arrays (local_sparsisty_.row_idxs and col_idxs) in row
     ** major ordering and the permutation index (local_sparsity_.ldu_mapping),
     ** which are required to create to a ginkgo matrix
     **/
    void init_local_sparsity_pattern(
        const lduInterfaceFieldPtrsList &interfaces) const;

    /** Based on OpenFOAMs interfaces this function computes two
     ** consecutive index arrays (non_local_sparsisty_.row_idxs and col_idxs) in
     ** row *major ordering and the permutation index
     ** (non_local_sparsity_.ldu_mapping), which are required to create a ginkgo
     ** matrix
     **/
    void init_non_local_sparsity_pattern(
        const lduInterfaceFieldPtrsList &interfaces) const;

    // updater

    /** Copies the LDU matrix coefficients to local_coeffs without changing or
     ** reinstantiating the sparsity pattern.
     ** This uses the local_sparsity_.ldu_mapping to permute the data already
     ** on the host to be in row major order
     **/
    void update_local_matrix_data(
        const lduInterfaceFieldPtrsList &interfaces,
        const FieldField<Field, scalar> &interfaceBouCoeffs) const;

    /** Copies the interface matrix coefficients to non_local_coeffs without
    ** changing or reinstantiating the sparsity pattern.
    **/
    void update_non_local_matrix_data(
        const lduInterfaceFieldPtrsList &interfaces,
        const FieldField<Field, scalar> &interfaceBouCoeffs) const;

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
        const dictionary &solverControls, const word &fieldName);

    // TODO reimplement this at some point
    // coupled wrapper constructor
    HostMatrixWrapper(const objectRegistry &db, const MatrixType &matrix,
                      const dictionary &solverControls, const word &fieldName);


    // getter

public:
    bool get_verbose() const { return verbose_; }

    const CommunicationPattern &get_communication_pattern() const
    {
        return communication_pattern_;
    };

    label get_local_nrows() const { return nrows_; }

    const ExecutorHandler &get_exec_handler() const { return exec_; }

    PersistentArray<scalar> &get_local_coeffs() const { return local_coeffs_; };

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
