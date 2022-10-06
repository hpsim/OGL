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

#include "HostMatrix.H"

#include "lduMatrix.H"


namespace Foam {

template <class MatrixType>
label HostMatrixWrapper<MatrixType>::compute_non_local_nnz(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    label ctr{0};
    for (int i = 0; i < interfaces.size(); i++) {
        if (interfaces.operator()(i) == nullptr) {
            continue;
        }
        const auto iface{interfaces.operator()(i)};
        if (isA<processorLduInterface>(iface->interface())) {
            ctr += iface->interface().faceCells().size();
        }
    }
    return ctr;
}

template label HostMatrixWrapper<lduMatrix>::compute_non_local_nnz(
    const lduInterfaceFieldPtrsList &interfaces_) const;

template <class MatrixType>
std::vector<scalar> HostMatrixWrapper<MatrixType>::communicate_non_local_coeffs(
    const lduInterfaceFieldPtrsList &interfaces,
    const FieldField<Field, scalar> &interfaceBouCoeffs) const
{
    std::vector<scalar> ret{};
    ret.reserve(nnz_non_local_matrix_);

    label startOfRequests = Pstream::nRequests();
    for (int i = 0; i < interfaces.size(); i++) {
        if (interfaces.operator()(i) == nullptr) {
            continue;
        }

        const auto iface{interfaces.operator()(i)};
        const auto &face_cells{interfaceBouCoeffs[i]};

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

            word msg_2 = "receive face cells interface " + std::to_string(i) +
                         " from proc " + std::to_string(neighbProcNo);
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

template std::vector<scalar>
HostMatrixWrapper<lduMatrix>::communicate_non_local_coeffs(
    const lduInterfaceFieldPtrsList &, const FieldField<Field, scalar> &) const;

// TODO merge with communicate_non_local_row_indices
template <class MatrixType>
std::vector<std::tuple<label, label, label>>
HostMatrixWrapper<MatrixType>::communicate_non_local_col_indices(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    // vector of local cell ids on other side
    std::vector<std::tuple<label, label, label>> non_local_idxs{};
    non_local_idxs.reserve(nnz_non_local_matrix_);

    label startOfRequests = Pstream::nRequests();
    for (int i = 0; i < interfaces.size(); i++) {
        if (interfaces.operator()(i) == nullptr) {
            continue;
        }

        const auto iface{interfaces.operator()(i)};
        const auto &face_cells{iface->interface().faceCells()};

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

    label interface_ctr = 0;
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

            word msg_2 = "receive face cells interface " + std::to_string(i) +
                         " from proc " + std::to_string(neighbProcNo);
            LOG_2(verbose_, msg_2)

            auto otherSide_tmp = pldui.receive<label>(
                Pstream::commsTypes::nonBlocking, interface_size);
            LOG_2(verbose_, "receive face cells done")

            for (label cellI = 0; cellI < interface_size; cellI++) {
                non_local_idxs.push_back(
                    {interface_ctr, face_cells[cellI],
                     global_row_index_.toGlobal(neighbProcNo,
                                                otherSide_tmp()[cellI])});
                interface_ctr += 1;
            }
        }
    }
    word msg = "done collecting neighbouring processor cell id";

    LOG_2(verbose_, msg)
    return non_local_idxs;
}

template std::vector<std::tuple<label, label, label>>
HostMatrixWrapper<lduMatrix>::communicate_non_local_col_indices(
    const lduInterfaceFieldPtrsList &interfaces) const;

template <class MatrixType>
void HostMatrixWrapper<MatrixType>::init_non_local_sparsity_pattern(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    auto non_local_row_indices = communicate_non_local_col_indices(interfaces);
    auto rows = non_local_sparsity_.row_idxs_.get_data();
    auto cols = non_local_sparsity_.col_idxs_.get_data();
    auto permute = non_local_sparsity_.ldu_mapping_.get_data();

    std::sort(non_local_row_indices.begin(), non_local_row_indices.end(),
              [&](const auto &a, const auto &b) {
                  auto [interface_idx_a, row_a, col_a] = a;
                  auto [interface_idx_b, row_b, col_b] = b;
                  return std::tie(row_a, col_a) < std::tie(row_b, col_b);
              });

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

            // check if current cell is on the current patch
            // NOTE cells can be several times on same patch
            for (label cellI = 0; cellI < interface_size; cellI++) {
                auto [interface_idx, row, col] =
                    non_local_row_indices[interface_ctr];
                rows[interface_ctr] = row;
                cols[interface_ctr] = col;
                permute[interface_ctr] = interface_idx;
                interface_ctr += 1;
            }
        }
    }
}

template void HostMatrixWrapper<lduMatrix>::init_non_local_sparsity_pattern(
    const lduInterfaceFieldPtrsList &interface) const;

template <class MatrixType>
void HostMatrixWrapper<MatrixType>::init_local_sparsity_pattern() const
{
    LOG_1(verbose_, "start init host sparsity pattern")

    auto lower_local = idx_array::view(
        exec_.get_ref_exec(), upper_nnz_,
        const_cast<label *>(&this->matrix().lduAddr().lowerAddr()[0]));

    auto upper_local = idx_array::view(
        exec_.get_ref_exec(), upper_nnz_,
        const_cast<label *>(&this->matrix().lduAddr().upperAddr()[0]));

    // row of upper, col of lower
    const auto lower = lower_local.get_const_data();

    // col of upper, row of lower
    const auto upper = upper_local.get_const_data();

    auto rows = local_sparsity_.row_idxs_.get_data();
    auto cols = local_sparsity_.col_idxs_.get_data();

    label element_ctr = 0;
    label upper_ctr = 0;

    std::vector<std::vector<std::pair<label, label>>> lower_stack(upper_nnz_);

    const auto permute = local_sparsity_.ldu_mapping_.get_data();

    // Scan through given rows and insert row and column indices into array
    //
    //  position after all local offdiagonal elements, needed for
    //  permutation matrix
    label after_neighbours = 2 * upper_nnz_;
    for (label row = 0; row < nrows_; row++) {
        // add lower elements
        // for now just scan till current upper ctr
        for (const auto [stored_upper_ctr, col] : lower_stack[row]) {
            rows[element_ctr] = row;
            cols[element_ctr] = col;
            permute[element_ctr] = stored_upper_ctr + upper_nnz_;
            element_ctr++;
        }

        // add diagonal elements
        rows[element_ctr] = row;
        cols[element_ctr] = row;
        permute[element_ctr] = after_neighbours + row;
        element_ctr++;

        // add upper elements
        // these are the transpose of the lower elements which are stored in
        // row major order.
        label row_upper = lower[upper_ctr];
        while (upper_ctr < upper_nnz_ && row_upper == row) {
            const label col_upper = upper[upper_ctr];

            rows[element_ctr] = row_upper;
            cols[element_ctr] = col_upper;

            // insert into lower_stack
            lower_stack[col_upper].emplace_back(upper_ctr, row_upper);
            permute[element_ctr] = upper_ctr;

            element_ctr++;
            upper_ctr++;
            row_upper = lower[upper_ctr];
        }
    }
    LOG_1(verbose_, "done init host matrix")
}

template void HostMatrixWrapper<lduMatrix>::init_local_sparsity_pattern() const;

template <class MatrixType>
void HostMatrixWrapper<MatrixType>::update_local_matrix_data() const
{
    auto ref_exec = exec_.get_ref_exec();

    // create a vector to hold contiguos matrix coefficients which can be
    // permuted from ldu to row major format ordering
    auto contiguos = vec::create(
        ref_exec,
        gko::dim<2>((gko::dim<2>::dimension_type)nnz_local_matrix_, 1));

    // copy upper
    auto upper = this->matrix().upper();
    auto u_host_view =
        gko::array<scalar>::view(ref_exec, upper_nnz_, &upper[0]);
    auto u_device_view =
        gko::array<scalar>::view(ref_exec, upper_nnz_, contiguos->get_values());
    u_device_view = u_host_view;

    // copy lower
    auto lower = this->matrix().lower();
    auto l_device_view = gko::array<scalar>::view(
        ref_exec, upper_nnz_, &contiguos->get_values()[upper_nnz_]);
    if (lower == upper) {
        // symmetric case reuse data already on the device
        l_device_view = u_device_view;

    } else {
        // non-symmetric case copy data to the device
        auto l_host_view =
            gko::array<scalar>::view(ref_exec, upper_nnz_, &lower[0]);
        l_device_view = l_host_view;
    }

    // copy diag
    auto diag = this->matrix().diag();
    auto diag_host_view = gko::array<scalar>::view(ref_exec, nrows_, &diag[0]);
    auto diag_contiguous_view = gko::array<scalar>::view(
        ref_exec, nrows_, &contiguos->get_values()[2 * upper_nnz_]);
    diag_contiguous_view = diag_host_view;

    auto dense_vec = vec::create(
        ref_exec,
        gko::dim<2>{(gko::dim<2>::dimension_type)nnz_local_matrix_, 1},
        gko::array<scalar>::view(ref_exec, nnz_local_matrix_,
                                 local_coeffs_.get_data()),
        1);

    const auto permute = local_sparsity_.ldu_mapping_.get_data();
    auto dense = dense_vec->get_values();
    auto contiguos_values = contiguos->get_values();
    for (label i = 0; i < nnz_local_matrix_; ++i) {
        dense[i] = contiguos_values[permute[i]];
    }
}

template void HostMatrixWrapper<lduMatrix>::update_local_matrix_data() const;

template <class MatrixType>
void HostMatrixWrapper<MatrixType>::update_non_local_matrix_data(
    const lduInterfaceFieldPtrsList &interfaces,
    const FieldField<Field, scalar> &interfaceBouCoeffs) const
{
    auto ref_exec = exec_.get_ref_exec();

    auto interface_coeffs =
        communicate_non_local_coeffs(interfaces, interfaceBouCoeffs);
    auto permute = non_local_sparsity_.ldu_mapping_.get_data();

    // copy interfaces
    auto tmp_contiguous_iface =
        gko::array<scalar>(ref_exec, nnz_non_local_matrix_);
    auto contiguous_iface = tmp_contiguous_iface.get_data();

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
            contiguous_iface[interface_ctr] =
                -interface_coeffs[permute[interface_ctr]];
            interface_ctr += 1;
        }
    }

    // copy to persistent
    auto i_device_view = gko::array<scalar>::view(
        ref_exec, nnz_non_local_matrix_, non_local_coeffs_.get_data());
    i_device_view = tmp_contiguous_iface;
}

template void HostMatrixWrapper<lduMatrix>::update_non_local_matrix_data(
    const lduInterfaceFieldPtrsList &, const FieldField<Field, scalar> &) const;
}  // namespace Foam
