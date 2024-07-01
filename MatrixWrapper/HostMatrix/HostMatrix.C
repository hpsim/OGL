// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "HostMatrix.H"

#include "cyclicFvPatchField.H"
#include "lduMatrix.H"
#include "processorFvPatch.H"

#include <map>

namespace Foam {


HostMatrixWrapper::HostMatrixWrapper(
    const ExecutorHandler &exec, const objectRegistry &db, label nrows,
    label upper_nnz, bool symmetric, const scalar *diag, const scalar *upper,
    const scalar *lower, const lduAddressing &addr,
    const FieldField<Field, scalar> &interfaceBouCoeffs,
    const FieldField<Field, scalar> &interfaceIntCoeffs,
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
      local_interface_nnz_{count_interface_nnz(interfaces, false)},
      upper_nnz_(upper_nnz),
      symmetric_(symmetric),
      non_diag_nnz_(2 * upper_nnz_),
      local_matrix_nnz_(nrows_ + 2 * upper_nnz_),
      local_matrix_w_interfaces_nnz_(local_matrix_nnz_ + local_interface_nnz_),
      interfaces_(interfaces),
      interfaceBouCoeffs_(interfaceBouCoeffs),
      non_local_matrix_nnz_{count_interface_nnz(interfaces, true)}
{}


label HostMatrixWrapper::count_interface_nnz(
    const lduInterfaceFieldPtrsList &interfaces, bool proc_interfaces) const
{
    label ctr{0};
    for (label i = 0; i < interfaces.size(); i++) {
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

std::vector<scalar> HostMatrixWrapper::collect_interface_coeffs(
    const lduInterfaceFieldPtrsList &interfaces,
    const FieldField<Field, scalar> &interfaceBouCoeffs, const bool local) const
{
    std::vector<scalar> ret{};
    ret.reserve((local) ? local_interface_nnz_ : non_local_matrix_nnz_);

    for (label i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }
        const auto iface{interface_getter(interfaces, i)};

        bool collect = (local)
                           ? !isA<processorLduInterface>(iface->interface())
                           : !!isA<processorLduInterface>(iface->interface());

        if (collect) {
            ret.insert(ret.end(), interfaceBouCoeffs[i].begin(),
                       interfaceBouCoeffs[i].end());
        }
    }

    std::for_each(ret.begin(), ret.end(), [](scalar &c) { c = c * -1.0; });

    return ret;
}

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


/** Same as interface_iterator but checks if is *NOT* Sel
 */
template <class Sel, class Func>
void neg_interface_iterator(const lduInterfaceFieldPtrsList &interfaces,
                            Func func)
{
    label element_ctr = 0;
    for (label i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }
        const auto iface{interface_getter(interfaces, i)};
        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();

        if (!isA<Sel>(iface->interface())) {
            func(element_ctr, interface_size, iface);
        }
    }
}


CommunicationPattern HostMatrixWrapper::create_communication_pattern() const
{
    using comm_size_type = CommunicationPattern::comm_size_type;
    // temp map, mapping from neighbour rank interface cells
    std::map<label, std::vector<label>> interface_cell_map{};

    // iterate all interfaces, count number of neighbour procs
    // and store rows to send to neighbour procs
    gko::size_type n_procs = 0;
    interface_iterator<processorFvPatch>(
        interfaces_, [&](label, label, label, const processorFvPatch &patch,
                         const lduInterfaceField *iface) {
            const auto &face_cells{iface->interface().faceCells()};
            const label neighbProcNo = patch.neighbProcNo();

            auto search = interface_cell_map.find(neighbProcNo);
            if (search == interface_cell_map.end()) {
                n_procs++;
                interface_cell_map.insert(std::pair{
                    neighbProcNo,
                    std::vector<label>(face_cells.begin(), face_cells.end())});
            } else {
                auto &vec = search->second;
                vec.insert(vec.end(), face_cells.begin(), face_cells.end());
            }
        });

    // create index_sets
    std::vector<std::pair<gko::array<label>, comm_size_type>> send_idxs;
    for (auto [proc, interface_cells] : interface_cell_map) {
        auto exec = exec_.get_ref_exec();
        send_idxs.push_back(std::pair<gko::array<label>, comm_size_type>(
            gko::array<label>(exec, interface_cells.begin(),
                              interface_cells.end()),
            proc));
    }

    // convert to gko::array
    gko::array<comm_size_type> target_ids{exec_.get_ref_exec(), n_procs};
    gko::array<comm_size_type> target_sizes{exec_.get_ref_exec(), n_procs};

    label iter = 0;
    for (const auto &[proc, interface_cells] : interface_cell_map) {
        target_ids.get_data()[iter] = proc;
        target_sizes.get_data()[iter] = interface_cells.size();
        iter++;
    }

    return CommunicationPattern{get_exec_handler(), target_ids, target_sizes, send_idxs};
}


template <typename PatchType>
void collect_local_interface_indices_impl(
    label &element_ctr, const lduInterfaceField *iface,
    const lduAddressing &addr,
    std::vector<std::tuple<label, label, label>> &local_interface_idxs)
{
    if (isA<PatchType>(iface->interface())) {
        const PatchType &patch = refCast<const PatchType>(iface->interface());
        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();
#ifdef WITH_ESI_VERSION
        const label neighbPatchId = patch.neighbPatchID();
#else
        const label neighbPatchId = patch.nbrPatchID();
#endif
        const labelUList &cols = addr.patchAddr(neighbPatchId);
        for (label cellI = 0; cellI < interface_size; cellI++) {
            local_interface_idxs.push_back(
                {element_ctr, face_cells[cellI], cols[cellI]});
            element_ctr += 1;
        }
    }
}

void collect_local_interface_indices_impl_cyclicAMIFvPatch(
    label &element_ctr, const lduInterfaceField *iface,
    const lduAddressing &addr,
    std::vector<std::tuple<label, label, label>> &local_interface_idxs)
{
    if (isA<cyclicAMIFvPatch>(iface->interface())) {
        FatalErrorInFunction
            << "Currently unsupported CyclicAMIFvPatch detected"
            << exit(FatalError);
        const cyclicAMIFvPatch &patch =
            refCast<const cyclicAMIFvPatch>(iface->interface());
        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();
#ifdef WITH_ESI_VERSION
        const label neighbPatchId = patch.cyclicAMIPatch().neighbPatchID();
#else
        const label neighbPatchId = patch.cyclicAMIPatch().nbrPatchID();
#endif
        const labelUList &cols = addr.patchAddr(neighbPatchId);
        for (label cellI = 0; cellI < interface_size; cellI++) {
            local_interface_idxs.push_back(
                {element_ctr, face_cells[cellI], cols[cellI]});
            element_ctr += 1;
        }
    }
}

#ifdef WITH_ESI_VERSION
void collect_local_interface_indices_impl_cyclicACMIFvPatch(
    label &element_ctr, const lduInterfaceField *iface,
    const lduAddressing &addr,
    std::vector<std::tuple<label, label, label>> &local_interface_idxs)
{
    if (isA<cyclicACMIFvPatch>(iface->interface())) {
        FatalErrorInFunction
            << "Currently unsupported CyclicACMIFvPatch detected"
            << exit(FatalError);
        const cyclicACMIFvPatch &patch =
            refCast<const cyclicACMIFvPatch>(iface->interface());
        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();
        const labelUList &cols =
            addr.patchAddr(patch.cyclicACMIPatch().neighbPatchID());
        for (label cellI = 0; cellI < interface_size; cellI++) {
            local_interface_idxs.push_back(
                {element_ctr, face_cells[cellI], cols[cellI]});
            element_ctr += 1;
        }
    }
}
#endif

std::vector<std::tuple<label, label, label>>
HostMatrixWrapper::collect_local_interface_indices(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    std::vector<std::tuple<label, label, label>> local_interface_idxs{};
    local_interface_idxs.reserve(local_interface_nnz_);

    neg_interface_iterator<processorFvPatch>(
        interfaces, [&](label &element_ctr, const label interface_size,
                        const lduInterfaceField *iface) {
            // check whether interface is either an cyclicFvPatch,
            // cyclicAMIFvPatch or cyclicACMIFvPatch and collect local interface
            // indices
            collect_local_interface_indices_impl<cyclicFvPatch>(
                element_ctr, iface, addr_, local_interface_idxs);
            collect_local_interface_indices_impl_cyclicAMIFvPatch(
                element_ctr, iface, addr_, local_interface_idxs);
#ifdef WITH_ESI_VERSION
            collect_local_interface_indices_impl_cyclicACMIFvPatch(
                element_ctr, iface, addr_, local_interface_idxs);
#endif
        });
    return local_interface_idxs;
}

std::vector<std::tuple<label, label, label, label>>
HostMatrixWrapper::collect_cells_on_non_local_interface(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    // vector of neighbour cell idx connected to interface
    std::vector<std::tuple<label, label, label, label>> non_local_idxs{};
    non_local_idxs.reserve(non_local_matrix_nnz_);
    interface_iterator<
        processorFvPatch>(interfaces, [&](label, label interface_id,
                                          label interface_size,
                                          const processorLduInterface &,
                                          const lduInterfaceField *iface) {
        const auto &face_cells{iface->interface().faceCells()};

        const processorLduInterface &pldui =
                refCast<const processorLduInterface>(iface->interface());
        const label neighbProcNo = pldui.neighbProcNo();
        pldui.send(Pstream::commsTypes::blocking, face_cells);

        auto otherSide_tmp = pldui.receive<label>(
                Pstream::commsTypes::blocking, interface_size);

        for (label cellI = 0; cellI < interface_size; cellI++) {
            auto local_row = face_cells[cellI];
            auto col = otherSide_tmp()[cellI];
            non_local_idxs.push_back({interface_id, col, local_row, neighbProcNo});
        }
    });

    word msg = "done collecting neighbouring processor cell id";
    LOG_2(verbose_, msg)
    return non_local_idxs;
}

std::shared_ptr<SparsityPattern> HostMatrixWrapper::compute_non_local_sparsity(
    std::shared_ptr<const gko::Executor> exec) const
{
    auto sparsity{std::make_shared<SparsityPattern>(exec->get_master(),
                                                    non_local_matrix_nnz_)};
    sparsity->dim = gko::dim<2>{nrows_, non_local_matrix_nnz_};

    auto non_local_indices =
        collect_cells_on_non_local_interface(interfaces_);
    auto rows = sparsity->row_idxs.get_data();
    auto cols = sparsity->col_idxs.get_data();
    auto permute = sparsity->ldu_mapping.get_data();
    label prev_interface_ctr{0};
    label end{0};
    label start{0};

    label element_ctr = 0;
    label interface_ctr{0};
    label prev_rank{0};

    // TODO currently we set permute eventhough this is not required
    // anymore, remove permute from non_local_interfaces
    for (auto [interface_idx, col, row, rank] : non_local_indices) {
        rows[element_ctr] = row;
        cols[element_ctr] = col;
        permute[element_ctr] = element_ctr;

        // a new interface started or the last element on last interface has been reached
        bool last_element = element_ctr == non_local_indices.size() - 1;
        if (interface_idx > prev_interface_ctr || last_element) {
            // this check will be reached one element earlier if we reached
            // the end of the non_local_indices thus we need to increment
            // the elment_ctr once more
            end = (last_element) ? element_ctr + 1 : element_ctr;
            sparsity->interface_spans.emplace_back(start, end);
            sparsity->rank.emplace_back(prev_rank);
            start = end;
            prev_interface_ctr = interface_idx;
        }
        prev_rank = rank;
        element_ctr++;
    }

    return sparsity;
}

std::shared_ptr<SparsityPattern> HostMatrixWrapper::compute_local_sparsity(
    std::shared_ptr<const gko::Executor> exec) const
{
    LOG_1(verbose_, "start init host sparsity pattern")

    auto sparsity{std::make_shared<SparsityPattern>(
        exec->get_master(), local_matrix_w_interfaces_nnz_)};

    auto lower_local = idx_array::view(
        exec, upper_nnz_, const_cast<label *>(addr_.lowerAddr().begin()));

    // TODO const_view ?
    auto upper_local = idx_array::view(
        exec, upper_nnz_, const_cast<label *>(addr_.upperAddr().begin()));

    // row of upper, col of lower
    const auto lower = lower_local.get_const_data();
    // col of upper, row of lower
    const auto upper = upper_local.get_const_data();

    auto rows = sparsity->row_idxs.get_data();
    auto cols = sparsity->col_idxs.get_data();
    const auto permute = sparsity->ldu_mapping.get_data();

    // Scan through given rows and insert row and column indices into array
    //
    // position after all local offdiagonal elements, needed for
    // permutation matrix
    //
    // TODO in order to simplify when local interfaces exists set
    // local_sparsity to size of nrows_w_interfaces, if interfaces exist
    // local_sparsity is only valid till nrows_
    label after_neighbours = (symmetric_) ? upper_nnz_ : 2 * upper_nnz_;
    init_local_sparsity(nrows_, upper_nnz_, symmetric_, upper, lower, rows,
                        cols, permute);

    sparsity->dim = gko::dim<2>{nrows_, nrows_};
    sparsity->interface_spans.emplace_back(0, local_matrix_nnz_);

    // if no local interfaces are present we are done here
    // otherwise we need to add local interfaces to local_sparsity in order
    // of the interfaces to end of the col and row idx arrays. This will produce
    // idx = [d_1, u_1, l_2, d_2, u_2, ... d_n, i_11, i_12, .., i_nn] where
    // i_j,k j=interface index and k cell index on the interface
    if (local_interface_nnz_) {
        // NOTE currently, this copies the interface indizes first to a vector
        // of tuples before inserting it into the persistent arrays. We could
        // remove the unnessary copy via the vector of tuples and
        // let collect_local_interface_indices_impl write directly to rows,
        // cols, permute etc
        auto local_interfaces = collect_local_interface_indices(interfaces_);

        label local_interface_ctr{0};
        for (label i = local_matrix_nnz_; i < local_matrix_w_interfaces_nnz_;
             ++i) {
            auto [interface_idx, interface_row, interface_col] =
                local_interfaces[local_interface_ctr];
            rows[i] = interface_row;
            cols[i] = interface_col;
            // if interfaces are treated separately we don't need to permute
            // interface values
            permute[i] = i;
            local_interface_ctr++;
        }

        // TODO merge with above
        local_interface_ctr = 0;
        label prev_interface_idx{0};
        label end{0};
        label start{local_matrix_nnz_};
        for (auto [interface_idx, interface_row, interface_col] :
             local_interfaces) {
            // check if a new interface has started or final interface has been
            // reache
            if (interface_idx > prev_interface_idx ||
                local_interface_ctr + 1 == local_interfaces.size()) {
                end = start + local_interface_ctr;
                local_interface_ctr = 0;
                sparsity->interface_spans.emplace_back(start, end);
                start = end;
                prev_interface_idx = interface_idx;
            }
            local_interface_ctr++;
        }
    }

    LOG_1(verbose_, "done init host sparsity pattern")
    return sparsity;
}

}  // namespace Foam
