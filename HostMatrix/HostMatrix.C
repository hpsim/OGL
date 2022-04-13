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
label HostMatrixWrapper<MatrixType>::count_elements_on_interfaces(
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

template label HostMatrixWrapper<lduMatrix>::count_elements_on_interfaces(
    const lduInterfaceFieldPtrsList &interfaces_) const;

// TODO merge with get_other_proc_cell_ids
template <class MatrixType>
std::vector<scalar> HostMatrixWrapper<MatrixType>::get_other_proc_bou_coeffs(
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
HostMatrixWrapper<lduMatrix>::get_other_proc_bou_coeffs(
    const label nInterfaces, const lduInterfaceFieldPtrsList &interfaces,
    const FieldField<Field, scalar> interfaceBouCoeffs);

// TODO merge with get_other_proc_cell_ids
template <class MatrixType>
std::vector<label> HostMatrixWrapper<MatrixType>::get_other_proc_cell_ids(
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

            word msg_2 = "receive face cells interface " + std::to_string(i) +
                         " from proc " + std::to_string(neighbProcNo);
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

template std::vector<label>
HostMatrixWrapper<lduMatrix>::get_other_proc_cell_ids(
    const label nInterfaces, const lduInterfaceFieldPtrsList &interfaces);

// TODO this is pretty much the same as get_other_proc_cell_ids
// and could probably trimmed down a lot
template <class MatrixType>
void HostMatrixWrapper<MatrixType>::insert_interface_coeffs(
    const lduInterfaceFieldPtrsList &interfaces,
    const std::vector<label> &other_proc_cell_ids, int *rows, int *cols,
    label row, label global_row, label &element_ctr, label *sorting_idxs,
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
            const label interface_idx = nElems_wo_Interfaces_ + interface_ctr;
            // check if current cell is on the current patch
            // NOTE cells can be several times on same patch
            for (label cellI = 0; cellI < interface_size; cellI++) {
                if (face_cells[cellI] == row) {
                    const label other_side_global_cellID =
                        global_cell_index_.toGlobal(
                            neighbProcNo,
                            other_proc_cell_ids[interface_ctr + cellI]);

                    rows[element_ctr] = global_row;
                    cols[element_ctr] = other_side_global_cellID;

                    sorting_idxs[element_ctr] = interface_idx + cellI;

                    element_ctr++;
                }
            }
            interface_ctr += interface_size;
        }
    }
}

template void HostMatrixWrapper<lduMatrix>::insert_interface_coeffs(
    const lduInterfaceFieldPtrsList &interfaces,
    const std::vector<label> &other_proc_cell_ids, int *rows, int *cols,
    label row, label global_row, label &element_ctr,
    label *sorting_interface_idxs, const bool upper) const;

template <class MatrixType>
void HostMatrixWrapper<MatrixType>::init_host_sparsity_pattern(
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

    // row of upper, col of lower
    const auto lower = lower_local.get_const_data();

    // col of upper, row of lower
    const auto upper = upper_local.get_const_data();

    auto rows = row_idxs_.get_data();  // row_idxs_local->get_data();
    auto cols = col_idxs_.get_data();  // col_idxs_local->get_data();

    label element_ctr = 0;
    label upper_ctr = 0;

    std::vector<std::vector<std::pair<label, label>>> lower_stack(nNeighbours_);


    const auto permute = ldu_csr_idx_mapping_.get_data();

    label interface_elem_ctr{0};
    label after_neighbours = 2 * nNeighbours_;
    for (label row = 0; row < nCells_; row++) {
        const label global_row = global_cell_index_.toGlobal(row);
        // check for lower idxs
        insert_interface_coeffs(interfaces, other_proc_cell_ids, rows, cols,
                                row, global_row, element_ctr, permute, true);

        // add lower elements
        // for now just scan till current upper ctr
        for (const auto [stored_upper_ctr, global_col] : lower_stack[row]) {
            rows[element_ctr] = global_row;
            cols[element_ctr] = global_col;
            permute[element_ctr] = stored_upper_ctr + nNeighbours_;
            element_ctr++;
        }

        // add diagonal elements
        rows[element_ctr] = global_row;
        cols[element_ctr] = global_row;
        permute[element_ctr] = after_neighbours + row;
        element_ctr++;

        // add upper elements
        // these are the transpose of the lower elements which are stored in row
        // major order.
        label row_upper = lower[upper_ctr];
        while (upper_ctr < nNeighbours_ && row_upper == row) {
            const label global_row_upper =
                global_cell_index_.toGlobal(row_upper);
            const label col_upper = upper[upper_ctr];
            const label global_col_upper =
                global_cell_index_.toGlobal(col_upper);

            rows[element_ctr] = global_row_upper;
            cols[element_ctr] = global_col_upper;

            // insert into lower_stack
            lower_stack[col_upper].emplace_back(upper_ctr, global_row_upper);
            permute[element_ctr] = upper_ctr;

            element_ctr++;
            upper_ctr++;
            row_upper = lower[upper_ctr];
        }

        insert_interface_coeffs(interfaces, other_proc_cell_ids, rows, cols,
                                row, global_row, element_ctr, permute, false);
    }
    LOG_1(verbose_, "done init host matrix")
}

template void HostMatrixWrapper<lduMatrix>::init_host_sparsity_pattern(
    const lduInterfaceFieldPtrsList &interfaces,
    const std::vector<label> other_proc_cell_ids) const;

template <class MatrixType>
void HostMatrixWrapper<MatrixType>::update_host_matrix_data(
    const lduInterfaceFieldPtrsList &interfaces,
    const std::vector<scalar> &interfaceBouCoeffs) const
{
    auto ref_exec = exec_.get_ref_exec();

    const auto sorting_idxs = ldu_csr_idx_mapping_.get_array();
    auto device_exec = exec_.get_device_exec();

    if (!permutation_stored_) {
        auto &db = values_.get_db();

        P_ = gko::share(gko::matrix::Permutation<label>::create(
            device_exec, gko::dim<2>{nElems_}, *sorting_idxs.get()));
        const fileName path = permutation_matrix_name_;
        auto po = new DevicePersistentBase<gko::LinOp>(IOobject(path, db), P_);
    }
    // END TODO

    // unsorted entries on device
    auto d = vec::create(device_exec, gko::dim<2>(nElems_, 1));

    // copy upper
    auto upper = this->matrix().upper();
    auto u_host_view =
        gko::Array<scalar>::view(ref_exec, nNeighbours_, &upper[0]);
    auto u_device_view =
        gko::Array<scalar>::view(device_exec, nNeighbours_, d->get_values());
    u_device_view = u_host_view;

    // copy lower
    auto lower = this->matrix().lower();
    auto l_device_view = gko::Array<scalar>::view(
        device_exec, nNeighbours_, &d->get_values()[nNeighbours_]);
    if (lower == upper) {
        // symmetric case reuse data already on the device
        l_device_view = u_device_view;

    } else {
        // non-symmetric case copy data to the device
        auto l_host_view =
            gko::Array<scalar>::view(ref_exec, nNeighbours_, &lower[0]);
        l_device_view = l_host_view;
    }

    // copy diag
    auto diag = this->matrix().diag();
    auto d_host_view = gko::Array<scalar>::view(ref_exec, nCells_, &diag[0]);
    auto d_device_view = gko::Array<scalar>::view(
        device_exec, nCells_, &d->get_values()[2 * nNeighbours_]);
    d_device_view = d_host_view;

    // copy interfaces
    auto tmp_contiguous_iface = gko::Array<scalar>(ref_exec, nInterfaces_);
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
            contiguous_iface[interface_ctr + cellI] =
                -interfaceBouCoeffs[interface_ctr + cellI];
        }
        interface_ctr += patch_size;
    }

    auto i_device_view = gko::Array<scalar>::view(
        device_exec, nInterfaces_, &d->get_values()[nElems_wo_Interfaces_]);
    i_device_view = tmp_contiguous_iface;


    auto dense_vec = vec::create(device_exec, gko::dim<2>{nElems_, 1});

    // NOTE apply changes the underlying pointer of dense_vec
    // thus copy_from is used to move the ptr to the underlying
    // device persistent array
    P_->apply(d.get(), dense_vec.get());

    auto dense_vec_after = vec::create(
        device_exec, gko::dim<2>{nElems_, 1},
        gko::Array<scalar>::view(device_exec, nElems_, values_.get_data()), 1);

    dense_vec_after->copy_from(dense_vec.get());
}

template void HostMatrixWrapper<lduMatrix>::update_host_matrix_data(
    const lduInterfaceFieldPtrsList &interfaces,
    const std::vector<scalar> &interfaceBouCoeffs) const;

}  // namespace Foam
