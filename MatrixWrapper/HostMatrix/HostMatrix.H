// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"
#include "processorLduInterface.H"


#include "DevicePersistent/Array/Array.H"
#include "DevicePersistent/DeviceIdGuard/DeviceIdGuard.H"
#include "DevicePersistent/IOGlobalIndex/gkoGlobalIndex.H"
#include "MatrixWrapper/SparsityPattern/SparsityPattern.H"
#include "CommunicationPattern/CommunicationPattern.H"
#include "common/common.H"


namespace Foam {

// Free functions
const lduInterfaceField *interface_getter(
    const lduInterfaceFieldPtrsList &interfaces, const label i);

/** Write contiguous row and col indices from OpenFOAM lower and upper indices
 ** For details on the lower triangular based indexing see
 ** https://openfoamwiki.net/index.php/OpenFOAM_guide/Matrices_in_OpenFOA
 ** Note that the order of the indices are depicted wrong on the wiki
 ** In general the upper triangular matrix is traversed in row major
 ** and the lower triangular matrix in column major order.
 **
 ** @param nrows number of rows
 ** @param upper_nnz number of non zeros in the upper triangular matrix
 ** @param is_symmetric whether matrix is symmetric, in the symmetric case
 ** the lower elements indices in the permute array are computed differently
 ** @param upper pointer to OFs rows array
 ** @param lower pointer to OFs cols array
 ** @param rows pointer to rows array
 ** @param cols pointer to columns array
 ** @param permute pointer to permuter array
 */
void init_local_sparsity(const label nrows, const label upper_nnz,
                         const bool is_symmetric, const label *upper,
                         const label *lower, label *rows, label *cols,
                         label *permute);



/* The HostMatrixWrapper class is a thin wrapper to simplify acess 
 * to OpenFOAM LDU matrix 
 * */
class HostMatrixWrapper {
private:
    using vec = gko::matrix::Dense<scalar>;
    using idx_array = gko::array<label>;

    const ExecutorHandler exec_;

    const DeviceIdGuardHandler device_id_guard_;

    const label verbose_;

    const word field_name_;

    // Whether the matrix coefficients should be reordered
    // during copy or on device
    const bool reorder_on_copy_;

    const lduAddressing &addr_;

    const scalar *diag_;

    const scalar *upper_;

    const scalar *lower_;

    // multiply the complete system by this factor, ie sAx=sb
    // NOTE this needed to avoid negative diagonal matrix entries, but
    // this could be also achieved by just fliping the sign
    const scalar scaling_;

    // number of local matrix rows
    const label nrows_;

    // number of local upper elements
    // ie coefficients which column_idx < nrows_
    const label upper_nnz_;

    const bool symmetric_;

    // number of local elements on interfaces
    // ie number of interfaces which column_idx is < nrows_
    // and have to be sorted into local matrix
    const label local_interface_nnz_;

    // total number of local upper and lower elements
    // ie 2*upper_nnz_ since the sparsity pattern is symmetric
    const label non_diag_nnz_;

    // nnz of local matrix wo local interfaces
    const label local_matrix_nnz_;

    // nnzs of local matrix including local interfaces
    const label local_matrix_w_interfaces_nnz_;

    // TODO remove
    // mutable SparsityPattern local_sparsity_pattern_;

    // non-local indices
    const label non_local_matrix_nnz_;

    const lduInterfaceFieldPtrsList &interfaces_;

    const FieldField<Field, scalar> &interfaceBouCoeffs_;

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


    /** Iterates all interfaces and collect the corresponding cell id (row)
    ** and a unique counter
    **
    ** @return vector of size nnz_non_local, with a running index, row
    ** index, and a unique compressed column index
    ** sections for each interface
    ** ret = [(1,2,1),(2,20, 2), (3, 20, 2) ...]
    **         i0   i1,   i...
    */
    std::vector<std::tuple<label, label, label, label>>
    collect_cells_on_non_local_interface(
        const lduInterfaceFieldPtrsList &interfaces) const;


    // updater


public:
    // segregated matrix wrapper constructor
    HostMatrixWrapper(const ExecutorHandler &exec, const objectRegistry &db,
                      label nrows, label upper_nnz, bool symmetric,
                      const scalar *diag, const scalar *upper,
                      const scalar *lower, const lduAddressing &addr,
                      const FieldField<Field, scalar> &interfaceBouCoeffs,
                      const FieldField<Field, scalar> &interfaceIntCoeffs,
                      const lduInterfaceFieldPtrsList &interfaces,
                      const dictionary &solverControls, const word &fieldName,
                      label verbose);

    // getter

public:
    /** Based on OpenFOAMs ldu matrix format this function computes two
     ** consecutive index arrays (local_sparsisty_.row_idxs and col_idxs) in row
     ** major ordering and the permutation index (local_sparsity_.ldu_mapping),
     ** which are required to create to a ginkgo matrix
     **/
    std::shared_ptr<SparsityPattern> compute_local_sparsity(
        std::shared_ptr<const gko::Executor> exec) const;

    /** Copies the LDU matrix coefficients to local_coeffs without changing or
     ** reinstantiating the sparsity pattern.
     ** This uses the local_sparsity_.ldu_mapping to permute the data already
     ** on the host or device to be in row major order.
     **/
    void compute_local_coeffs(std::shared_ptr<const SparsityPattern> sparsity,
                              gko::array<scalar> &target_coeffs) const;

    /** Copies the interface matrix coefficients to non_local_coeffs without
    ** changing or reinstantiating the sparsity pattern.
    **/
    void compute_non_local_coeffs(
        std::shared_ptr<const SparsityPattern> sparsity,
        gko::array<scalar> &target_coeffs) const;

    /** Based on OpenFOAMs interfaces this function computes two
     ** consecutive index arrays (non_local_sparsisty_.row_idxs and col_idxs) in
     ** row *major ordering and the permutation index
     ** (non_local_sparsity_.ldu_mapping), which are required to create a ginkgo
     ** matrix
     **/
    std::shared_ptr<SparsityPattern> compute_non_local_sparsity(
        std::shared_ptr<const gko::Executor> exec) const;

    /** Iterates all interfaces and counts the number of unique neighbour
     ** processors and number of interfaces in total for this processor
     ** and collects all interface cells of this rank.
     **
     ** @param interfaces The list of interfaces for the search
     ** @return the CommunicationPattern
     */
    CommunicationPattern create_communication_pattern() const;

    bool get_verbose() const { return verbose_; }

    label get_local_nrows() const { return nrows_; }

    gko::dim<2> get_size() const { return gko::dim<2>(nrows_, nrows_); }

    const ExecutorHandler &get_exec_handler() const { return exec_; }
};


}  // namespace Foam
