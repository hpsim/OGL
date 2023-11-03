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

#include "cyclicFvPatchField.H"
#include "lduMatrix.H"


namespace Foam {

const lduInterfaceField *interface_getter(
    const lduInterfaceFieldPtrsList &interfaces, const label i)
{
#ifdef WITH_ESI_VERSION
    return interfaces.get(i);
#else
    return interfaces.operator()(i);
#endif
}

template <class MatrixType>
label HostMatrixWrapper<MatrixType>::count_interface_nnz(
    const lduInterfaceFieldPtrsList &interfaces, bool proc_interfaces) const
{
    label ctr{0};
    for (int i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }
        const auto iface{interface_getter(interfaces, i)};

        bool count = (proc_interfaces)
                         ? !!isA<processorLduInterface>(iface->interface())
                         : !isA<processorLduInterface>(iface->interface());
        if (count) {
            ctr += iface->interface().faceCells().size();
        }
    }
    return ctr;
}

template label HostMatrixWrapper<lduMatrix>::count_interface_nnz(
    const lduInterfaceFieldPtrsList &interfaces, bool proc_interfaces) const;


template <class MatrixType>
std::vector<scalar> HostMatrixWrapper<MatrixType>::collect_interface_coeffs(
    const lduInterfaceFieldPtrsList &interfaces,
    const FieldField<Field, scalar> &interfaceBouCoeffs, const bool local) const
{
    std::vector<scalar> ret{};
    ret.reserve((local) ? local_interface_nnz_ : non_local_matrix_nnz_);

    for (int i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }
        const auto iface{interface_getter(interfaces, i)};
        auto coeffs{interfaceBouCoeffs[i]};

        bool collect = (local)
                           ? !isA<processorLduInterface>(iface->interface())
                           : !!isA<processorLduInterface>(iface->interface());

        if (collect) {
            const label interface_size = iface->interface().faceCells().size();
            for (label cellI = 0; cellI < interface_size; cellI++) {
                ret.push_back(coeffs[cellI]);
            }
        }
    }

    return ret;
}


template std::vector<scalar>
HostMatrixWrapper<lduMatrix>::collect_interface_coeffs(
    const lduInterfaceFieldPtrsList &, const FieldField<Field, scalar> &,
    const bool) const;

template <class MatrixType>
std::vector<std::tuple<label, label, label>>
HostMatrixWrapper<MatrixType>::collect_local_interface_indices(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    std::vector<std::tuple<label, label, label>> local_interface_idxs{};
    local_interface_idxs.reserve(local_interface_nnz_);

    for (int i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }

        const auto iface{interface_getter(interfaces, i)};

        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();

        // TODO make this a separate specialized function
        label interface_ctr = 0;
        if (isA<cyclicLduInterface>(iface->interface())) {
            const cyclicLduInterface &pldui =
                refCast<const cyclicLduInterface>(iface->interface());
            // const labelUList &rows = this->matrix().lduAddr().patchAddr(i);

#ifdef WITH_ESI_VERSION
            const label neighbPatchId = pldui.neighbPatchID();
#else
            const label neighbPatchId = pldui.nbrPatchID();
#endif
            const labelUList &cols =
                this->matrix().lduAddr().patchAddr(neighbPatchId);
            for (label cellI = 0; cellI < interface_size; cellI++) {
                // DEBUG
                // cout << "patch_ctr " << cellI << " ictr " << interface_ctr
                //     << " patchId " << i << " neighPatchId " << neighbPatchId
                //      << " row " << rows[cellI] << " col " << cols[cellI]
                //      << "\n" << endl;
                local_interface_idxs.push_back(
                    {interface_ctr, face_cells[cellI], cols[cellI]});
                interface_ctr += 1;
            }
        }
    }
    return local_interface_idxs;
}

template <class MatrixType>
std::pair<gko::array<label>, gko::array<label>>
HostMatrixWrapper<MatrixType>::assemble_proc_id_and_sizes(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    // temp vector to store neighbour proc number and number of cells to send
    std::vector<std::pair<label, label>> neighbour_procs{};
    // TODO FIXME use a lambda to avoid repeating
    for (int i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }

        const auto &iface{interface_getter(interfaces, i)};
        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();

        if (isA<processorLduInterface>(iface->interface())) {
            const processorLduInterface &pldui =
                refCast<const processorLduInterface>(iface->interface());
            const label neighbProcNo = pldui.neighbProcNo();

            neighbour_procs.push_back(
                std::pair<label, label>{neighbProcNo, interface_size});
        }
    }

    // reduce vector
    label n_procs = 0;
    std::map<label, label> reduce_map{};
    for (auto [proc, n_faces] : neighbour_procs) {
        auto search = reduce_map.find(proc);
        if (search == reduce_map.end()) {
            n_procs += 1;
            reduce_map.insert(std::pair<label,label>{proc, n_faces});
        } else {
            reduce_map[proc] = reduce_map[proc] + n_faces;
        }
    }

    // convert to gko::array
    gko::array<label> target_ids{exec_.get_ref_exec(), n_procs};
    gko::array<label> target_sizes{exec_.get_ref_exec(), n_procs};

    label iter = 0;
    for (const auto &[proc, size] : reduce_map) {
        target_ids.get_data()[iter] = proc;
        target_sizes.get_data()[iter] = size;
        iter++;
    }

    return std::pair(target_ids, target_sizes);
}

template
std::pair<gko::array<label>, gko::array<label>>
HostMatrixWrapper<lduMatrix>::assemble_proc_id_and_sizes(
    const lduInterfaceFieldPtrsList &interfaces) const;

template <class MatrixType>
std::vector<std::tuple<label, label, label>>
HostMatrixWrapper<MatrixType>::collect_non_local_col_indices(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    // vector of local cell ids on other side
    std::vector<std::tuple<label, label, label>> non_local_idxs{};
    non_local_idxs.reserve(non_local_matrix_nnz_);

    label interface_ctr = 0;
    for (int i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }

        const auto iface{interface_getter(interfaces, i)};
        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();

        if (isA<processorLduInterface>(iface->interface())) {
            for (label cellI = 0; cellI < interface_size; cellI++) {
                non_local_idxs.push_back(
                    {interface_ctr, face_cells[cellI], interface_ctr});
                interface_ctr += 1;
            }
        }
    }
    word msg = "done collecting neighbouring processor cell id";

    LOG_2(verbose_, msg)
    return non_local_idxs;
}

template std::vector<std::tuple<label, label, label>>
HostMatrixWrapper<lduMatrix>::collect_non_local_col_indices(
    const lduInterfaceFieldPtrsList &interfaces) const;

template <class MatrixType>
void HostMatrixWrapper<MatrixType>::init_non_local_sparsity_pattern(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    auto non_local_row_indices = collect_non_local_col_indices(interfaces);
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
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }

        const auto &iface{interface_getter(interfaces, i)};
        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();

        if (isA<processorLduInterface>(iface->interface())) {
            // const processorLduInterface &pldui =
            //     refCast<const processorLduInterface>(iface->interface());

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
    const lduInterfaceFieldPtrsList &interfaces) const;

template <class MatrixType>
void HostMatrixWrapper<MatrixType>::init_local_sparsity_pattern(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    LOG_1(verbose_, "start init host sparsity pattern")
    bool is_symmetric{this->matrix().symmetric()};

    auto lower_local = idx_array::view(
        exec_.get_ref_exec(), upper_nnz_ - local_interface_nnz_,
        const_cast<label *>(&this->matrix().lduAddr().lowerAddr()[0]));

    auto upper_local = idx_array::view(
        exec_.get_ref_exec(), upper_nnz_ - local_interface_nnz_,
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
    //
    //  TODO in order to simplify when local interfaces exists set
    //  local_sparsity to size of nrows_w_interfaces, if interfaces exist
    //  local_sparsity is only valid till nrows_
    label after_neighbours = (is_symmetric) ? upper_nnz_ : 2 * upper_nnz_;
    for (label row = 0; row < nrows_; row++) {
        // add lower elements
        // for now just scan till current upper ctr
        for (const auto &[stored_upper_ctr, col] : lower_stack[row]) {
            rows[element_ctr] = row;
            cols[element_ctr] = col;
            permute[element_ctr] = stored_upper_ctr;
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
            lower_stack[col_upper].emplace_back(
                (is_symmetric) ? upper_ctr : upper_ctr + upper_nnz_, row_upper);
            permute[element_ctr] = upper_ctr;

            element_ctr++;
            upper_ctr++;
            row_upper = lower[upper_ctr];
        }
    }

    // if no local interfaces are present we are done here
    // otherwise we need to add local interfaces in order
    if (local_interface_nnz_) {
        auto local_interfaces = collect_local_interface_indices(interfaces);
        // // sort local interfaces to be in row major order
        // // and keep original indices
        std::sort(local_interfaces.begin(), local_interfaces.end(),
                  [&](const auto &a, const auto &b) {
                      auto [interface_idx_a, row_a, col_a] = a;
                      auto [interface_idx_b, row_b, col_b] = b;
                      return std::tie(row_a, col_a) < std::tie(row_b, col_b);
                  });

        // copy current rows and columns to tmp array
        // TODO this has full length but is only valid till nrows_
        auto rows_copy_a = gko::array<label>(
            *local_sparsity_.row_idxs_.get_persistent_object().get());
        auto cols_copy_a = gko::array<label>(
            *local_sparsity_.col_idxs_.get_persistent_object().get());
        auto permute_copy_a = gko::array<label>(
            *local_sparsity_.ldu_mapping_.get_persistent_object().get());

        auto rows_copy = rows_copy_a.get_data();
        auto cols_copy = cols_copy_a.get_data();
        auto permute_copy = permute_copy_a.get_data();

        auto rows = local_sparsity_.row_idxs_.get_data();
        auto cols = local_sparsity_.col_idxs_.get_data();
        auto permute = local_sparsity_.ldu_mapping_.get_data();

        label current_idx_ctr = 0;
        label total_ctr = 0;
        // iterate local interfaces i
        // insert all coeffs for row < interface[i].row
        // find local_interface with lowes row, column
        for (auto const &interface : local_interfaces) {
            auto [interface_idx, interface_row, interface_col] = interface;

            // copy from existing matrix coefficients
            while ([&]() {
                // check for length
                if (current_idx_ctr == local_matrix_nnz_) {
                    return false;
                }

                if (rows_copy[current_idx_ctr] > interface_row) {
                    return false;
                }

                // the copy rows are or equal
                // in that case we need to check if the
                // copy columns are lower
                if (rows_copy[current_idx_ctr] == interface_row &&
                    cols_copy[current_idx_ctr] > interface_col) {
                    return false;
                }
                return true;
            }()) {
                // insert coeffs
                rows[total_ctr] = rows_copy[current_idx_ctr];
                cols[total_ctr] = cols_copy[current_idx_ctr];
                permute[total_ctr] = permute_copy[current_idx_ctr];
                current_idx_ctr++;
                total_ctr++;
            }
            rows[total_ctr] = interface_row;
            cols[total_ctr] = interface_col;
            // store the original position of the contiguous interfaces
            permute[total_ctr] = after_neighbours + nrows_ + interface_idx;
            total_ctr++;
        }

        // post insert if local interfaces were consumed but stuff remains in
        // rows_copy and cols_copy
        for (int i = total_ctr; i < local_matrix_w_interfaces_nnz_; i++) {
            // std::cout << "insert missing values " << i << endl;
            rows[i] = rows_copy[current_idx_ctr];
            cols[i] = cols_copy[current_idx_ctr];
            permute[i] = permute_copy[current_idx_ctr];
            current_idx_ctr++;
        }
    }

    LOG_1(verbose_, "done init host matrix")
}

template void HostMatrixWrapper<lduMatrix>::init_local_sparsity_pattern(
    const lduInterfaceFieldPtrsList &interfaces) const;

template <class MatrixType>
void HostMatrixWrapper<MatrixType>::update_local_matrix_data(
    const lduInterfaceFieldPtrsList &interfaces,
    const FieldField<Field, scalar> &interfaceBouCoeffs) const
{
    auto ref_exec = exec_.get_ref_exec();
    auto upper = this->matrix().upper();
    auto lower = this->matrix().lower();
    auto diag = this->matrix().diag();
    label diag_nnz = diag.size();
    bool is_symmetric{this->matrix().symmetric()};

    auto dense_vec = vec::create(
        ref_exec,
        gko::dim<2>{(gko::dim<2>::dimension_type)local_matrix_nnz_, 1},
        gko::array<scalar>::view(ref_exec, local_matrix_nnz_,
                                 local_coeffs_.get_data()),
        1);

    // TODO this does not work for Ell
    const auto permute = local_sparsity_.ldu_mapping_.get_data();
    auto dense = dense_vec->get_values();

    if (local_interface_nnz_) {
        auto couple_coeffs =
            collect_interface_coeffs(interfaces, interfaceBouCoeffs, true);
        if (is_symmetric) {
            for (label i = 0; i < local_matrix_w_interfaces_nnz_; ++i) {
                // where the element is stored in a combined array
                const label pos{permute[i]};
                scalar value;
                // all values up to upper_nnz_ are upper_nnz_ values
                if (pos < upper_nnz_) {
                    value = upper[pos];
                }
                // all values up to upper_nnz_ are upper_nnz_ values
                if (pos >= upper_nnz_ && pos < upper_nnz_ + diag_nnz) {
                    value = diag[pos - upper_nnz_];
                }
                if (pos >= upper_nnz_ + diag_nnz) {
                    // std::cout <<  "idx ( "
                    // << local_sparsity_.row_idxs_.get_data()[i] << ", "
                    // << local_sparsity_.col_idxs_.get_data()[i] << "): "
                    // << value << "\n" ;
                    value = -couple_coeffs[pos - upper_nnz_ - diag_nnz];
                }

                dense[i] = scaling_ * value;
            }
        } else {
            for (label i = 0; i < local_matrix_w_interfaces_nnz_; ++i) {
                const label pos{permute[i]};
                scalar value;
                if (pos < upper_nnz_) {
                    value = upper[pos];
                }
                if (pos >= upper_nnz_ && pos < 2 * upper_nnz_) {
                    value = lower[pos - upper_nnz_];
                }
                if (pos >= 2 * upper_nnz_ && pos < 2 * upper_nnz_ + diag_nnz) {
                    value = diag[pos - 2 * upper_nnz_];
                }
                if (pos >= 2 * upper_nnz_ + diag_nnz) {
                    value = -couple_coeffs[pos - 2 * upper_nnz_ - diag_nnz];
                }
                dense[i] = scaling_ * value;
            }
        }
    } else {
        // TODO move this to a function fill_non_interfaces
        if (is_symmetric) {
            for (label i = 0; i < local_matrix_nnz_; ++i) {
                const label pos{permute[i]};
                dense[i] = scaling_ * (pos >= upper_nnz_)
                               ? diag[pos - upper_nnz_]
                               : upper[pos];
            }
            return;
        } else {
            for (label i = 0; i < local_matrix_nnz_; ++i) {
                const label pos{permute[i]};
                if (pos < upper_nnz_) {
                    dense[i] = scaling_ * upper[pos];
                    continue;
                }
                if (pos >= upper_nnz_ && pos < 2 * upper_nnz_) {
                    dense[i] = scaling_ * lower[pos - upper_nnz_];
                    continue;
                }
                dense[i] = scaling_ * diag[pos - 2 * upper_nnz_];
            }
        }
    }
}

template void HostMatrixWrapper<lduMatrix>::update_local_matrix_data(
    const lduInterfaceFieldPtrsList &interfaces,
    const FieldField<Field, scalar> &interfaceBouCoeffs) const;


template <class MatrixType>
void HostMatrixWrapper<MatrixType>::update_non_local_matrix_data(
    const lduInterfaceFieldPtrsList &interfaces,
    const FieldField<Field, scalar> &interfaceBouCoeffs) const
{
    auto ref_exec = exec_.get_ref_exec();

    auto interface_coeffs =
        collect_interface_coeffs(interfaces, interfaceBouCoeffs, false);
    auto permute = non_local_sparsity_.ldu_mapping_.get_data();

    // copy interfaces
    auto tmp_contiguous_iface =
        gko::array<scalar>(ref_exec, non_local_matrix_nnz_);
    auto contiguous_iface = tmp_contiguous_iface.get_data();

    label interface_ctr{0};
    for (int i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }
        const auto iface{interface_getter(interfaces, i)};
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
        ref_exec, non_local_matrix_nnz_, non_local_coeffs_.get_data());
    i_device_view = tmp_contiguous_iface;
}

template void HostMatrixWrapper<lduMatrix>::update_non_local_matrix_data(
    const lduInterfaceFieldPtrsList &, const FieldField<Field, scalar> &) const;
}  // namespace Foam
