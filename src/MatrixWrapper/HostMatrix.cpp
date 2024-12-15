// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/HostMatrix.hpp"

#include "cyclicFvPatchField.H"
#include "lduMatrix.H"
#include "processorFvPatch.H"

#include <map>

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


HostMatrixWrapper::HostMatrixWrapper(
    const ExecutorHandler &exec, const objectRegistry &db, label nrows,
    label upper_nnz, bool symmetric, const scalar *diag, const scalar *upper,
    const scalar *lower, const lduAddressing &addr,
    const FieldField<Field, scalar> &interfaceBouCoeffs,
    [[maybe_unused]] const FieldField<Field, scalar> &interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList &interfaces,
    const dictionary &solverControls, const word &fieldName, label verbose)
    : exec_{exec},
      device_id_guard_{db, fieldName, exec_.get_device_exec()},
      verbose_(verbose),
      field_name_(fieldName),
      reorder_on_copy_(
          solverControls.lookupOrDefault<Switch>("reorderOnHost", true)),
      addr_(addr),
      diag_(diag),
      upper_(upper),
      lower_(lower),
      scaling_(solverControls.lookupOrDefault<scalar>("scaling", 1)),
      nrows_(nrows),
      upper_nnz_(upper_nnz),
      symmetric_(symmetric),
      non_diag_nnz_(2 * upper_nnz_),
      local_matrix_nnz_(nrows_ + 2 * upper_nnz_),
      interfaces_(interfaces),
      interfaceBouCoeffs_(interfaceBouCoeffs)
{
    auto ref_exec = exec.get_ref_exec();
    auto comm = *exec.get_communicator().get();
    label rank = exec.get_rank();

    using pair_dtype = std::pair<label, const scalar *>;
    // TODO this needs to be consistent with how sparsity are generated
    // so this should merged with sparsity generation
    // upper
    interface_ptr_.emplace(
        std::make_pair<label, pair_dtype>(0, {upper_nnz_, upper_}));
    // lower
    interface_ptr_.emplace(
        std::make_pair<label, pair_dtype>(1, {upper_nnz_, lower_}));
    // diag
    interface_ptr_.emplace(
        std::make_pair<label, pair_dtype>(2, {nrows_, diag}));

    // compute global interface idx
    label local_interface_cnt = interfaces.size();
    auto global_interfaces_recv = std::vector<label>(comm.size());

    comm.all_gather(ref_exec, &local_interface_cnt, 1,
                    global_interfaces_recv.data(), 1);


    size_t counter{0};
    local_to_global_interface_idx_.resize(global_interfaces_recv.size());
    for (size_t i = 0; i < global_interfaces_recv.size(); i++) {
        local_to_global_interface_idx_[i] = counter;
        counter += global_interfaces_recv[i];
    }

    for (label i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }
        const auto iface{interface_getter(interfaces, i)};
        auto interface_length{iface->interface().faceCells().size()};
        if (interface_length == 0) {
            continue;
        }
        label global_interface_id =
            (local_to_global_interface_idx_[rank] + i) * -1;
        interface_ptr_.emplace<label, pair_dtype>(
            std::move(global_interface_id),
            {interface_length, interfaceBouCoeffs[i].begin()});
    }
}

HostMatrixWrapper::HostMatrixWrapper(
    const ExecutorHandler &exec, const objectRegistry &db,
    const lduAddressing &addr, bool symmetric, const scalar *diag,
    const scalar *upper, const scalar *lower,
    const FieldField<Field, scalar> &interfaceBouCoeffs,
    const FieldField<Field, scalar> &interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList &interfaces,
    const dictionary &solverControls, const word &fieldName, label verbose)
    : HostMatrixWrapper::HostMatrixWrapper(
          exec, db, addr.size(), addr.lowerAddr().size(), symmetric, diag,
          upper, lower, addr, interfaceBouCoeffs, interfaceIntCoeffs,
          interfaces, solverControls, fieldName, verbose)
{}

template <class Sel, class Func>
void interface_iterator(const lduInterfaceFieldPtrsList &interfaces, Func func)
{
    label element_ctr = 0;
    label interface_ctr = 0;

    for (label i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }
        const auto iface{interface_getter(interfaces, i)};
        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();

        if (isA<Sel>(iface->interface())) {
            const Sel &patch = refCast<const Sel>(iface->interface());
            func(element_ctr, interface_ctr, interface_size, patch, iface);
            interface_ctr++;
        }
    }
}

std::shared_ptr<CommunicationPattern>
HostMatrixWrapper::create_communication_pattern() const
{
    using comm_size_type = CommunicationPattern::comm_size_type;
    // temp map, mapping from neighbour rank interface cells
    std::map<label, std::vector<label>> interface_cell_map{};
    std::vector<label> target_ids;

    // iterate all interfaces, count number of neighbour procs
    // and store rows to send to neighbour procs
    interface_iterator<processorFvPatch>(
        interfaces_, [&](label, label, label, const processorFvPatch &patch,
                         const lduInterfaceField *iface) {
            const auto &face_cells{iface->interface().faceCells()};
            const label neighbProcNo = patch.neighbProcNo();

            auto search = interface_cell_map.find(neighbProcNo);
            if (search == interface_cell_map.end()) {
                target_ids.push_back(neighbProcNo);
                interface_cell_map.insert(std::pair{
                    neighbProcNo,
                    std::vector<label>(face_cells.begin(), face_cells.end())});
            } else {
                std::vector<label> &neighbour_cells =
                    interface_cell_map[neighbProcNo];
                neighbour_cells.insert(neighbour_cells.end(),
                                       face_cells.begin(), face_cells.end());
            }
        });

    // create index_sets
    std::vector<std::vector<label>> send_idxs;
    for (label proc : target_ids) {
        send_idxs.emplace_back(interface_cell_map[proc]);
    }

    return std::make_shared<CommunicationPattern>(get_exec_handler(),
                                                  target_ids, send_idxs);
}

std::shared_ptr<SparsityPattern> HostMatrixWrapper::compute_interface_sparsity(
    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
        partition) const
{
    // vector of neighbour cell idx connected to interface
    auto rank = get_exec_handler().get_rank();
    auto pattern = std::make_shared<SparsityPattern>();

    for (label i = 0; i < interfaces_.size(); i++) {
        if (interface_getter(interfaces_, i) == nullptr) {
            continue;
        }
        const auto iface{interface_getter(interfaces_, i)};
        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();

        if (isA<processorFvPatch>(iface->interface())) {
            if (interface_size == 0) {
                continue;
            }


            const auto &coupledPatch =
                refCast<const coupledFvPatch>(iface->interface());
            const auto &patch =
                refCast<const processorFvPatch>(iface->interface());


            const processorLduInterface &pldui =
                refCast<const processorLduInterface>(iface->interface());
            const label neighbProcNo = pldui.neighbProcNo();
            pldui.send(Pstream::commsTypes::blocking, face_cells);

            auto other_side_tmp = pldui.receive<label>(
                Pstream::commsTypes::blocking, interface_size);
            auto cols =
                std::vector<label>(other_side_tmp->cdata(),
                                   other_side_tmp->cdata() + interface_size);

            pattern->insert_interface(
                std::vector<label>(face_cells.cdata(),
                                   face_cells.cdata() + interface_size),
                convert_to_global(partition, cols.data(), interface_size,
                                  neighbProcNo),
                rank, neighbProcNo, -(local_to_global_interface_idx_[rank] + i),
                -(local_to_global_interface_idx_[neighbProcNo] + i));
        }

        if (isA<cyclicFvPatch>(iface->interface())) {
            if (interface_size == 0) {
                continue;
            }

            const cyclicFvPatch &patch =
                refCast<const cyclicFvPatch>(iface->interface());
#ifdef WITH_ESI_VERSION
            const label neighbPatchId = patch.neighbPatchID();
#else
            const label neighbPatchId = patch.nbrPatchID();
#endif
            const labelUList &cols = addr_.patchAddr(neighbPatchId);

            // FIXME the other comm_id is wrong
            pattern->insert_interface(
                std::vector<label>(face_cells.cdata(),
                                   face_cells.cdata() + interface_size),
                std::vector<label>(cols.cdata(), cols.cdata() + interface_size),
                rank, rank, (i + 1) * -1, (i + 1) * -1);
        }
    }
    return pattern;
}


std::pair<std::shared_ptr<SparsityPattern>, std::shared_ptr<SparsityPattern>>
HostMatrixWrapper::compute_sparsity_patterns(
    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
        partition) const
{
    auto local_sparsity = compute_local_sparsity();
    auto non_local_sparsity = compute_interface_sparsity(partition);
    auto rank = get_exec_handler().get_rank();
    local_sparsity->move_interface(non_local_sparsity, rank,
                                   [](auto in) { return in; });
    return {local_sparsity, non_local_sparsity};
}

std::shared_ptr<SparsityPattern> HostMatrixWrapper::compute_local_sparsity()
    const
{
    LOG_1(verbose_, "start init host sparsity pattern")

    // row of upper, col of lower
    const auto lower = addr_.lowerAddr().begin();
    // col of upper, row of lower
    const auto upper = addr_.upperAddr().begin();

    auto pattern = std::make_shared<SparsityPattern>();

    label rank = Pstream::myProcNo();

    // insert upper
    pattern->insert_interface(std::vector(lower, lower + upper_nnz_),
                              std::vector(upper, upper + upper_nnz_), rank,
                              rank, 0, 0);
    // insert lower
    pattern->insert_interface(std::vector(upper, upper + upper_nnz_),
                              std::vector(lower, lower + upper_nnz_), rank,
                              rank, 1, 1, false);
    // insert diag
    std::vector<label> drows(nrows_);
    std::iota(drows.begin(), drows.end(), 0);
    std::vector<label> dcols(nrows_);
    std::iota(dcols.begin(), dcols.end(), 0);

    pattern->insert_interface(std::move(drows), std::move(dcols), rank, rank, 2,
                              2);

    // Scan through given rows and insert row and column indices into array
    //
    // position after all local offdiagonal elements, needed for
    // permutation matrix
    //
    // TODO in order to simplify when local interfaces exists set
    // local_sparsity to size of nrows_w_interfaces, if interfaces exist
    // local_sparsity is only valid till nrows_
    // init_local_sparsity(nrows_, upper_nnz_, upper, lower, rows, cols,
    // permute);

    // if no local interfaces are present we are done here
    // otherwise we need to add local interfaces to local_sparsity in order
    // of the interfaces to end of the col and row idx arrays. This will produce
    // idx = [d_1, u_1, l_2, d_2, u_2, ... d_n, i_11, i_12, .., i_nn] where
    // i_j,k j=interface index and k cell index on the interface

    LOG_1(verbose_, "done init host sparsity pattern")
    return pattern;
}

}  // namespace Foam
