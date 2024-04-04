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
#include "processorFvPatch.H"

#include <map>

namespace Foam {

template <class MatrixType>
HostMatrixWrapper<MatrixType>::HostMatrixWrapper(
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
      reorder_on_copy_(
          solverControls.lookupOrDefault<Switch>("reorderOnHost", false)),
      scaling_(solverControls.lookupOrDefault<scalar>("scaling", 1)),
      nrows_(matrix.diag().size()),
      local_interface_nnz_(count_interface_nnz(interfaces, false)),
      upper_nnz_(matrix.lduAddr().upperAddr().size()),
      non_diag_nnz_(2 * upper_nnz_),
      local_matrix_nnz_(nrows_ + 2 * upper_nnz_),
      local_matrix_w_interfaces_nnz_(local_matrix_nnz_ + local_interface_nnz_),
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
          true,              // needs to be updated
          !reorder_on_copy_  // whether to init on device
                             // if reorderOnHost is selected, the array needs
                             // to be initialized on the host
      },
      non_local_matrix_nnz_(count_interface_nnz(interfaces, true)),
      communication_pattern_(create_communication_pattern(interfaces)),
      non_local_sparsity_{
          fieldName + "_non_local", db, exec_, non_local_matrix_nnz_, verbose_,
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
      P_{(permutation_stored_)
             ? db.template lookupObjectRef<DevicePersistentBase<gko::LinOp>>(
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


template <class MatrixType>
HostMatrixWrapper<MatrixType>::HostMatrixWrapper(
    const objectRegistry &db, const MatrixType &matrix,
    const dictionary &solverControls, const word &fieldName)
    : MatrixType::solver(fieldName, matrix, solverControls),
      exec_{db, solverControls, fieldName},
      device_id_guard_{db, fieldName, exec_.get_device_exec()},
      verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
      reorder_on_copy_(
          solverControls.lookupOrDefault<Switch>("reorderOnHost", true)),
      scaling_(solverControls.lookupOrDefault<scalar>("scaling", 1)),
      nrows_(matrix.diag().size()),
      local_interface_nnz_(0),
      upper_nnz_(matrix.lduAddr().upperAddr().size()),
      non_diag_nnz_(2 * upper_nnz_),
      local_matrix_nnz_(nrows_ + 2 * upper_nnz_),
      local_matrix_w_interfaces_nnz_(local_matrix_nnz_ + local_interface_nnz_),
      local_sparsity_{
          fieldName + "_cols", db, exec_, local_matrix_nnz_, verbose_,
      },
      local_coeffs_{
          fieldName + "_coeffs",
          db,
          exec_,
          local_matrix_nnz_,
          verbose_,
          true,              // needs to be updated
          !reorder_on_copy_  // whether to init on device
                             // if reorderOnHost is selected, the array needs
                             // to be initialized on the host
      },
      non_local_matrix_nnz_(),
      // proc_target_id_and_sizes_(assemble_proc_id_and_sizes(interfaces)),
      non_local_sparsity_{
          fieldName + "_non_local", db, exec_, non_local_matrix_nnz_, verbose_,
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
      P_{(permutation_stored_)
             ? db.template lookupObjectRef<DevicePersistentBase<gko::LinOp>>(
                     permutation_matrix_name_)
                   .get_ptr()
             : nullptr}
{
    FatalErrorInFunction << "This constructor is currently not implemented"
                         << abort(FatalError);
}

template <class MatrixType>
label HostMatrixWrapper<MatrixType>::count_interface_nnz(
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

template <class MatrixType>
std::vector<scalar> HostMatrixWrapper<MatrixType>::collect_interface_coeffs(
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
    for (label i = 0; i < interfaces.size(); i++) {
        if (interface_getter(interfaces, i) == nullptr) {
            continue;
        }
        const auto iface{interface_getter(interfaces, i)};
        const auto &face_cells{iface->interface().faceCells()};
        const label interface_size = face_cells.size();

        if (isA<Sel>(iface->interface())) {
            const Sel &patch = refCast<const Sel>(iface->interface());
            func(element_ctr, interface_size, patch, iface);
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


template <class MatrixType>
CommunicationPattern
HostMatrixWrapper<MatrixType>::create_communication_pattern(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    using comm_size_type = CommunicationPattern::comm_size_type;
    // temp map, mapping from neighbour rank interface cells
    std::map<label, std::vector<label>> interface_cell_map{};

    // iterate all interfaces, count number of neighbour procs
    // and store rows to send to neighbour procs
    gko::size_type n_procs = 0;
    interface_iterator<processorFvPatch>(
        interfaces, [&](label, const label, const processorFvPatch &patch,
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

    return CommunicationPattern{target_ids, target_sizes, send_idxs};
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

template <class MatrixType>
std::vector<std::tuple<label, label, label>>
HostMatrixWrapper<MatrixType>::collect_local_interface_indices(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    std::vector<std::tuple<label, label, label>> local_interface_idxs{};
    local_interface_idxs.reserve(local_interface_nnz_);

    const auto &addr = this->matrix().lduAddr();
    neg_interface_iterator<processorFvPatch>(
        interfaces, [&](label &element_ctr, const label interface_size,
                        const lduInterfaceField *iface) {
            // check whether interface is either an cyclicFvPatch,
            // cyclicAMIFvPatch or cyclicACMIFvPatch and collect local interface
            // indices
            collect_local_interface_indices_impl<cyclicFvPatch>(
                element_ctr, iface, addr, local_interface_idxs);
            collect_local_interface_indices_impl_cyclicAMIFvPatch(
                element_ctr, iface, addr, local_interface_idxs);
#ifdef WITH_ESI_VERSION
            collect_local_interface_indices_impl_cyclicACMIFvPatch(
                element_ctr, iface, addr, local_interface_idxs);
#endif
        });
    return local_interface_idxs;
}

template <class MatrixType>
std::vector<std::tuple<label, label>>
HostMatrixWrapper<MatrixType>::collect_cells_on_non_local_interface(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    // vector of neighbour cell idx connected to interface
    std::vector<std::tuple<label, label>> non_local_idxs{};
    non_local_idxs.reserve(non_local_matrix_nnz_);
    label interface_ctr = 0;
    interface_iterator<processorFvPatch>(
        interfaces,
        [&](label, const label interface_size, const processorLduInterface &,
            const lduInterfaceField *iface) {
            const auto &face_cells{iface->interface().faceCells()};
            for (label cellI = 0; cellI < interface_size; cellI++) {
                auto local_row = face_cells[cellI];
                non_local_idxs.push_back({interface_ctr, local_row});
                interface_ctr += 1;
            }
        });

    word msg = "done collecting neighbouring processor cell id";
    LOG_2(verbose_, msg)
    return non_local_idxs;
}

template <class MatrixType>
void HostMatrixWrapper<MatrixType>::init_non_local_sparsity_pattern(
    const lduInterfaceFieldPtrsList &interfaces) const
{
    auto non_local_row_indices =
        collect_cells_on_non_local_interface(interfaces);
    auto rows = non_local_sparsity_.row_idxs_.get_data();
    auto cols = non_local_sparsity_.col_idxs_.get_data();
    auto permute = non_local_sparsity_.ldu_mapping_.get_data();

    // TODO if we treat all interfaces separately we don't need to sort
    // anymore
    //
    // Sorting of the interfaces is still needed since we can we have
    // multiple interfaces using the same rows/send_idxs
    // if the resulting non_local_matrix is not in row major order
    // we might get non converging solvers on GPU devices
    std::sort(non_local_row_indices.begin(), non_local_row_indices.end(),
              [&](const auto &a, const auto &b) {
                  auto [interface_idx_a, row_a] = a;
                  auto [interface_idx_b, row_b] = b;
                  return row_a < row_b;
              });

    label element_ctr = 0;
    for (auto [interface_idx, row] : non_local_row_indices) {
        rows[element_ctr] = row;
        cols[element_ctr] = interface_idx;
        permute[element_ctr] = interface_idx;
        element_ctr += 1;
    }
}

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
    const auto permute = local_sparsity_.ldu_mapping_.get_data();

    // Scan through given rows and insert row and column indices into array
    //
    // position after all local offdiagonal elements, needed for
    // permutation matrix
    //
    // TODO in order to simplify when local interfaces exists set
    // local_sparsity to size of nrows_w_interfaces, if interfaces exist
    // local_sparsity is only valid till nrows_
    label after_neighbours = (is_symmetric) ? upper_nnz_ : 2 * upper_nnz_;
    init_local_sparsity(nrows_, upper_nnz_, is_symmetric, upper, lower, rows,
                        cols, permute);

    // if no local interfaces are present we are done here
    // otherwise we need to add local interfaces to local_sparsity in order
    // of the interfaces to end of the col and row idx arrays. This will produce
    // idx = [d_1, u_1, l_2, d_2, u_2, ... d_n, i_11, i_12, .., i_nn] where
    // i_j,k j=interface index and k cell index on the interface
    if (local_interface_nnz_) {
        // NOTE currently, this copies the interface indizes first to a vector
        // of tuples before inserting it into the persistent arrays. We could
        // remove the unnessary copy via the vector of tuples and
        // let collect_local_interface_indices_impl write directly to rows, cols, permute etc
        auto local_interfaces = collect_local_interface_indices(interfaces);

        label local_interface_ctr {0};
        for (label i=local_matrix_nnz_; i<local_matrix_w_interfaces_nnz_ ;++i) {
          auto [interface_idx, interface_row, interface_col] = local_interfaces[local_interface_ctr];
          rows[i] = interface_row;
          cols[i] = interface_col;
          // if interfaces are treated separately we don't need to permute
          // interface values
          permute[i] = i;
          local_interface_ctr++;
        }
    }
    LOG_1(verbose_, "done init host matrix")
}


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

    // TODO this does not work for Ell
    auto permute = local_sparsity_.ldu_mapping_.get_data();

    if (reorder_on_copy_) {
        auto dense = local_coeffs_.get_data();
        if (local_interface_nnz_) {
            auto couple_coeffs =
                collect_interface_coeffs(interfaces, interfaceBouCoeffs, true);
            if (is_symmetric) {
                symmetric_update_w_interface(
                    local_matrix_w_interfaces_nnz_, diag_nnz, upper_nnz_,
                    permute, scaling_, diag.data(), upper.data(),
                    couple_coeffs.data(), dense);

            } else {
                non_symmetric_update_w_interface(
                    local_matrix_w_interfaces_nnz_, diag_nnz, upper_nnz_,
                    permute, scaling_, diag.data(), upper.data(), lower.data(),
                    couple_coeffs.data(), dense);
            }
        } else {
            if (is_symmetric) {
                symmetric_update(local_matrix_nnz_, upper_nnz_, permute,
                                 scaling_, diag.data(), upper.data(), dense);
            } else {
                non_symmetric_update(local_matrix_nnz_, upper_nnz_, permute,
                                     scaling_, diag.data(), upper.data(),
                                     lower.data(), dense);
            }
        }
    } else {
        // TODO
        // - this should be moved to separate function to avoid making this
        // too long

        using dim_type = gko::dim<2>::dimension_type;

        auto target_exec =
            (reorder_on_copy_) ? ref_exec : exec_.get_device_exec();

        // copy upper
        auto upper = this->matrix().upper();
        auto u_host_view =
            gko::array<scalar>::view(ref_exec, upper_nnz_, &upper[0]);
        auto u_device_view = gko::array<scalar>::view(target_exec, upper_nnz_,
                                                      local_coeffs_.get_data());
        u_device_view = u_host_view;

        // for the non-symmetric case copy lower to the device
        if (!is_symmetric) {
            auto lower = this->matrix().lower();
            auto l_host_view =
                gko::array<scalar>::view(ref_exec, upper_nnz_, &lower[0]);
            auto l_device_view = gko::array<scalar>::view(
                target_exec, upper_nnz_, &local_coeffs_.get_data()[upper_nnz_]);
            l_device_view = l_host_view;
        }

        // copy diag
        auto diag = this->matrix().diag();
        auto diag_host_view =
            gko::array<scalar>::view(ref_exec, nrows_, &diag[0]);
        const label diag_start = (is_symmetric) ? upper_nnz_ : 2 * upper_nnz_;
        auto diag_device_view = gko::array<scalar>::view(
            target_exec, nrows_, &local_coeffs_.get_data()[diag_start]);
        diag_device_view = diag_host_view;

        // copy local interfaces
        if (local_interface_nnz_) {
            const label interface_start = diag_start + nrows_;
            auto couple_coeffs =
                collect_interface_coeffs(interfaces, interfaceBouCoeffs, true);
            auto interface_host_view = gko::array<scalar>::view(
                ref_exec, local_interface_nnz_, couple_coeffs.data());
            auto interface_device_view = gko::array<scalar>::view(
                target_exec, local_interface_nnz_,
                &local_coeffs_.get_data()[interface_start]);
            interface_device_view = interface_host_view;
        }

        // permute and update local_coeffs
        auto row_collection = gko::share(gko::matrix::Dense<scalar>::create(
            target_exec,
            gko::dim<2>{static_cast<dim_type>(local_matrix_w_interfaces_nnz_),
                        1}));

        auto dense_vec = vec::create(
            target_exec,
            gko::dim<2>{static_cast<dim_type>(local_matrix_w_interfaces_nnz_),
                        1},
            gko::array<scalar>::view(target_exec,
                                     local_matrix_w_interfaces_nnz_,
                                     local_coeffs_.get_data()),
            1);

        // TODO this needs to copy ldu_mapping to the device
        dense_vec->row_gather(local_sparsity_.ldu_mapping_.get_array().get(),
                              row_collection.get());

        dense_vec->copy_from(row_collection);
    }

    // NOTE interface_spans and dim are not persistent so we need to
    // recreate this for every solver call
    // TODO this needs a proper implementation once we know how to handle
    // interfaces
    local_sparsity_.dim_ = gko::dim<2>{nrows_, nrows_};

    local_sparsity_.interface_spans_.emplace_back(0, local_matrix_nnz_);
    if (local_interface_nnz_) {
        auto local_interfaces = collect_local_interface_indices(interfaces);

        label local_interface_ctr {0};
        label prev_interface_ctr {0};
        label end = {0};
        label start = local_matrix_nnz_;
        for (auto [interface_idx, interface_row, interface_col] : local_interfaces){
          // A new interface started
          if (interface_idx > prev_interface_ctr) {
            end = start + i;
            local_sparsity_.interface_spans_.emplace_back(start, end);
            start = end;
            prev_interface_ctr = interface_idx;
          }
        }
    }
}


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

    for (label element_ctr = 0; element_ctr < non_local_matrix_nnz_;
         element_ctr++) {
        contiguous_iface[element_ctr] = interface_coeffs[permute[element_ctr]];
    }

    // copy to persistent
    auto i_device_view = gko::array<scalar>::view(
        ref_exec, non_local_matrix_nnz_, non_local_coeffs_.get_data());
    i_device_view = tmp_contiguous_iface;

    // NOTE interface_spans and dim are not persistent so we need to
    // We have now the full non_local matrix and need to add the start and end
    // to the PersistentSparsityPattern data structure
    non_local_sparsity_.interface_spans_.emplace_back(0, non_local_matrix_nnz_);
    non_local_sparsity_.dim_ = gko::dim<2>{nrows_, non_local_matrix_nnz_};
}

template HostMatrixWrapper<lduMatrix>::HostMatrixWrapper(
    const objectRegistry &db, const lduMatrix &matrix,
    // coeffs for cells on boundaries
    const FieldField<Field, scalar> &interfaceBouCoeffs,
    // coeffs for internal cells
    const FieldField<Field, scalar> &interfaceIntCoeffs,
    // pointers to interfaces can be used to access concrete
    // functions such as transferring indices, patch neighbours etc
    const lduInterfaceFieldPtrsList &interfaces,
    const dictionary &solverControls, const word &fieldName);

}  // namespace Foam
