// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/HostMatrix.H"

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

void init_local_sparsity(const label nrows, const label upper_nnz,
                         // const bool is_symmetric,
                         const label *upper, const label *lower, label *rows,
                         label *cols, label *permute)
{
    // for OpenFOAMs addressing see
    // https://openfoamwiki.net/index.php/OpenFOAM_guide/Matrices_in_OpenFOAM
    // Note that the face order in the wiki seems to be wrong. Entries are
    // stored such that upper rows are monotonic ascending
    // upper - rows of lower triangular matrix
    // lower - columns of lower triangular matrix
    // TODO FIXME
    // if symmetric we can reuse already copied data
    // see also note in Distributed.C
    label after_neighbours = 2 * upper_nnz;

    // first pass order elements row wise
    // scan through all faces
    std::vector<std::tuple<label, label, label>> tmp_upper;
    tmp_upper.reserve(upper_nnz);
    for (label faceI = 0; faceI < upper_nnz; faceI++) {
        const label col = upper[faceI];
        const label row = lower[faceI];
        tmp_upper.emplace_back(row, col, faceI);
    }

    std::sort(tmp_upper.begin(), tmp_upper.end(),
              [&](const auto &a, const auto &b) {
                  auto [row_a, col_a, faceI_a] = a;
                  auto [row_b, col_b, faceI_b] = b;
                  return std::tie(row_a, col_a) < std::tie(row_b, col_b);
              });

    std::vector<std::tuple<label, label, label>> tmp_lower;
    tmp_lower.reserve(upper_nnz);
    for (label faceI = 0; faceI < upper_nnz; faceI++) {
        const label col = lower[faceI];
        const label row = upper[faceI];
        tmp_lower.emplace_back(row, col, faceI);
    }

    std::sort(tmp_lower.begin(), tmp_lower.end(),
              [&](const auto &a, const auto &b) {
                  auto [row_a, col_a, faceI_a] = a;
                  auto [row_b, col_b, faceI_b] = b;
                  return std::tie(row_a, col_a) < std::tie(row_b, col_b);
              });

    // now we have tmp_upper and tmp_lower in row order
    label element_ctr = 0;
    label upper_ctr = 0;
    label lower_ctr = 0;
    label lower_size = tmp_lower.size();
    label upper_size = tmp_upper.size();
    auto [row_lower, col_lower, faceI_lower] = tmp_lower[0];
    auto [row_upper, col_upper, faceI_upper] = tmp_upper[0];
    for (label row = 0; row < nrows; row++) {
        // check if we have any lower elements to insert
        while (row_lower == row) {
            rows[element_ctr] = row_lower;
            cols[element_ctr] = col_lower;
            permute[element_ctr] = upper_nnz + faceI_lower;
            // TODO we copy the full vector to GPU thus this is not
            // required at the moment
            // permute[element_ctr] =
            //     (is_symmetric) ? faceI_lower : upper_nnz + faceI_lower;
            element_ctr++;
            lower_ctr++;
            if (lower_ctr >= lower_size) {
                break;
            }
            auto [tmp_row_lower, tmp_col_lower, tmp_faceI_lower] =
                tmp_lower[lower_ctr];
            row_lower = tmp_row_lower;
            col_lower = tmp_col_lower;
            faceI_lower = tmp_faceI_lower;
        }

        // add diagonal elements
        rows[element_ctr] = row;
        cols[element_ctr] = row;
        permute[element_ctr] = after_neighbours + row;
        element_ctr++;

        // check if we have any upper elements to insert
        while (row_upper == row) {
            rows[element_ctr] = row_upper;
            cols[element_ctr] = col_upper;
            permute[element_ctr] = faceI_upper;
            element_ctr++;
            upper_ctr++;
            if (upper_ctr >= upper_size) {
                break;
            }
            auto [tmp_row_upper, tmp_col_upper, tmp_faceI_upper] =
                tmp_upper[upper_ctr];
            row_upper = tmp_row_upper;
            col_upper = tmp_col_upper;
            faceI_upper = tmp_faceI_upper;
        }
    }
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
      local_interface_nnz_{count_interface_nnz(interfaces, false)},
      non_diag_nnz_(2 * upper_nnz_),
      local_matrix_nnz_(nrows_ + 2 * upper_nnz_),
      local_matrix_w_interfaces_nnz_(local_matrix_nnz_ + local_interface_nnz_),
      non_local_matrix_nnz_{count_interface_nnz(interfaces, true)},
      interfaces_(interfaces),
      interfaceBouCoeffs_(interfaceBouCoeffs)
{
    for (label i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }
        const auto iface{interface_getter(interfaces, i)};
        interface_length_.push_back(iface->interface().faceCells().size());
        interface_ptr_.push_back(interfaceBouCoeffs[i].begin());
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


std::shared_ptr<CommunicationPattern>
HostMatrixWrapper::create_communication_pattern() const
{
    using comm_size_type = CommunicationPattern::comm_size_type;
    // temp map, mapping from neighbour rank interface cells
    std::map<label, std::vector<label>> interface_cell_map{};
    std::vector<label> target_ids;

    // iterate all interfaces, count number of neighbour procs
    // and store rows to send to neighbour procs
    gko::size_type n_procs = 0;
    interface_iterator<processorFvPatch>(
        interfaces_, [&](label, label, label, const processorFvPatch &patch,
                         const lduInterfaceField *iface) {
            const auto &face_cells{iface->interface().faceCells()};
            const label neighbProcNo = patch.neighbProcNo();

            target_ids.push_back(neighbProcNo);

            auto search = interface_cell_map.find(neighbProcNo);
            if (search == interface_cell_map.end()) {
                n_procs++;
                interface_cell_map.insert(std::pair{
                    neighbProcNo,
                    std::vector<label>(face_cells.begin(), face_cells.end())});
            } else {
                std::vector<label>& neighbour_cells = interface_cell_map[neighbProcNo];
                neighbour_cells.insert(neighbour_cells.end(), face_cells.begin(), face_cells.end());
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
        interfaces,
        [&](label &element_ctr, [[maybe_unused]] const label interface_size,
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
    interface_iterator<processorFvPatch>(
        interfaces,
        [&](label, label interface_id, label interface_size,
            const processorLduInterface &, const lduInterfaceField *iface) {
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
                non_local_idxs.push_back(
                    {interface_id, col, local_row, neighbProcNo});
            }
        });

    word msg = "done collecting neighbouring processor cell id";
    LOG_2(verbose_, msg)
    return non_local_idxs;
}

std::shared_ptr<SparsityPattern> HostMatrixWrapper::compute_non_local_sparsity(
    std::shared_ptr<const gko::Executor> exec) const
{
    auto non_local_indices = collect_cells_on_non_local_interface(interfaces_);
    std::vector<label> rows_vec(non_local_matrix_nnz_);
    std::vector<label> cols_vec(non_local_matrix_nnz_);
    std::vector<label> mapping_vec(non_local_matrix_nnz_);
    std::vector<gko::span> spans{};

    auto rows = rows_vec.data();
    auto cols = cols_vec.data();
    auto permute = mapping_vec.data();
    label prev_interface_ctr{0};
    label end{0};
    label start{0};

    size_t element_ctr = 0;
    // label interface_ctr{0};

    for (auto [interface_idx, col, row, rank] : non_local_indices) {
        rows[element_ctr] = row;
        cols[element_ctr] = col;
        permute[element_ctr] = element_ctr;

        // a new interface started been reached
        if (interface_idx > prev_interface_ctr) {
            end = element_ctr;
            spans.emplace_back(start, element_ctr);
            start = end;
            prev_interface_ctr = interface_idx;
        }
        element_ctr++;
    }
    spans.emplace_back(end, non_local_indices.size());

    gko::dim<2> dim{static_cast<gko::size_type>(nrows_),
                    static_cast<gko::size_type>(non_local_matrix_nnz_)};
    return std::make_shared<SparsityPattern>(exec->get_master(), dim, rows_vec,
                                             cols_vec, mapping_vec, spans);
}


std::shared_ptr<SparsityPattern> HostMatrixWrapper::compute_local_sparsity(
    std::shared_ptr<const gko::Executor> exec) const
{
    LOG_1(verbose_, "start init host sparsity pattern")

    std::vector<label> rows_vec(local_matrix_w_interfaces_nnz_);
    std::vector<label> cols_vec(local_matrix_w_interfaces_nnz_);
    std::vector<label> mapping_vec(local_matrix_w_interfaces_nnz_);
    std::vector<gko::span> spans{gko::span{
        0, static_cast<gko::size_type>(local_matrix_w_interfaces_nnz_)}};

    auto rows = rows_vec.data();
    auto cols = cols_vec.data();
    auto permute = mapping_vec.data();

    auto lower_local = idx_array::view(
        exec, upper_nnz_, const_cast<label *>(addr_.lowerAddr().begin()));

    // TODO const_view ?
    auto upper_local = idx_array::view(
        exec, upper_nnz_, const_cast<label *>(addr_.upperAddr().begin()));

    // row of upper, col of lower
    const auto lower = lower_local.get_const_data();
    // col of upper, row of lower
    const auto upper = upper_local.get_const_data();

    // Scan through given rows and insert row and column indices into array
    //
    // position after all local offdiagonal elements, needed for
    // permutation matrix
    //
    // TODO in order to simplify when local interfaces exists set
    // local_sparsity to size of nrows_w_interfaces, if interfaces exist
    // local_sparsity is only valid till nrows_
    init_local_sparsity(nrows_, upper_nnz_, upper, lower, rows, cols, permute);

    // if no local interfaces are present we are done here
    // otherwise we need to add local interfaces to local_sparsity in order
    // of the interfaces to end of the col and row idx arrays. This will produce
    // idx = [d_1, u_1, l_2, d_2, u_2, ... d_n, i_11, i_12, .., i_nn] where
    // i_j,k j=interface index and k cell index on the interface
    if (local_interface_nnz_) {
        // NOTE currently, this copies the interface indizes first to a vector
        // of tuples before inserting it into the persistent arrays. We could
        // remove the unnecessary copy via the vector of tuples and
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
                static_cast<size_t>(local_interface_ctr + 1) ==
                    local_interfaces.size()) {
                end = start + local_interface_ctr;
                local_interface_ctr = 0;
                spans.emplace_back(start, end);
                start = end;
                prev_interface_idx = interface_idx;
            }
            local_interface_ctr++;
        }
    }

    LOG_1(verbose_, "done init host sparsity pattern")
    return std::make_shared<SparsityPattern>(
        exec->get_master(), get_size(), rows_vec, cols_vec, mapping_vec, spans);
}

}  // namespace Foam
