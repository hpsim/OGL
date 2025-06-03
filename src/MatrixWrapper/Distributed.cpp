// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/Distributed.hpp"
#include <fstream>

/* helper function to convert a sparsity pattern into a vector of linops with
 * zero coefficients
 *
 */
template <typename MatrixType>
std::vector<std::shared_ptr<gko::LinOp>> generate_inner_linops(
    const ExecutorHandler &exec_handler, gko::dim<2> dim,
    const std::vector<std::vector<label>> &rowss,
    const std::vector<std::vector<label>> &colss, const std::vector<label> &ids,
    std::map<label, scalar *> &linops, bool fuse)
{
    OGL_ASSERT_EQ(rowss.size(), colss.size());
    auto exec = exec_handler.get_device_exec();
    std::vector<std::shared_ptr<gko::LinOp>> lin_ops;

    bool needs_col_major = false;

    if constexpr (std::is_same_v<MatrixType, gko::matrix::Ell<scalar, label>>) {
        needs_col_major = true;
    }


    // create empty matrix if no interfaces are present
    if (rowss.size() == 0) {
        lin_ops.push_back(
            gko::share(MatrixType::create(exec, gko::dim<2>{0, 0})));
        return lin_ops;
    }
    size_t offset = 0;
    for (size_t i = 0; i < rowss.size(); i++) {
        const auto &rows = rowss[i];
        const auto &cols = colss[i];
        gko::array<scalar> coeffs(exec, rows.size());
        coeffs.fill(0.0);
        auto mtx_data = gko::device_matrix_data<scalar, label>(
            exec->get_master(), dim,
            gko::array<label>(exec->get_master(), rows.begin(), rows.end()),
            gko::array<label>(exec->get_master(), cols.begin(), cols.end()),
            coeffs);
        auto mtx = gko::share(MatrixType::create(exec));
        gko::as<MatrixType>(mtx)->read(mtx_data);
        linops[ids[i]] = mtx->get_values() + offset;
        offset += (fuse) ? rows.size() : 0;
        lin_ops.push_back(mtx);
    }
    return lin_ops;
}

// TODO let this function just dispatch
// to fused/unfused or owner/non_owner
// also can be cleaned up a lot
template <typename MatrixType>
void generate_pairwise_update_data(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const HostMatrixWrapper> host_A,
    std::shared_ptr<SparsityPattern> in,  // repartitioned_sparsity on owner
    bool fuse, std::shared_ptr<const Repartitioner> repartitioner,
    std::map<label, scalar *> linops, size_t start_offset,
    std::vector<RepartDistMatrix::pairwise_data> &update_data)
{
    // iterate interface data from sparsity pattern
    // need to find original id, whether this rank sends or receives, the
    // comm_rank, interface_size and pointer to send to
    bool owner = repartitioner->is_owner(exec_handler);
    label rank = exec_handler.get_host_rank();
    size_t offset = start_offset;

    for (size_t i = 0; i < in->get_id().size(); i++) {
        auto id = in->get_id()[i];
        auto orig_rank = in->get_orig_rank()[i];
        label length = in->get_rows()[i].size();
        label linop_id = -1;  //(fuse)? -1 : id;
                              //
        // skip id since id >= 0 are ldu and never needs pairwise
        // communication
        if (id >= 0) {
            continue;
        }

        label mode = -1;
        label comm_rank = -1;
        label send_id = id;

        bool local =
            repartitioner->get_owner_rank(in->get_comm_rank()[i]) == rank;

        linop_id = id;
        scalar *recv_ptr = nullptr;
        if (!owner) {
            mode = 0;  // send
            comm_rank = repartitioner->get_owner_rank(in->get_orig_rank()[i]);
        } else {
            // if already on rank mark as local otherwise always receive
            comm_rank = in->get_orig_rank()[i];
            bool relocated = comm_rank != rank;
            mode = (relocated) ? 1 : 2;
            send_id = (relocated) ? 0 : id;
            size_t get_id = linop_id;
            if (fuse) {
                get_id = (start_offset == 0) ? -1 : 0;
            }
            recv_ptr = linops[get_id] + offset;
            offset += (fuse) ? length : 0;
        }
        // std::cout << __FILE__ << __LINE__ << " set update_data rank " << rank
        //           << " comm_rank " << comm_rank << " id " << id << " length "
        //           << interface_length << " owner " << owner << " send data "
        //           << send_data_ptr << " recv_data_ptr " << recv_data_ptr
        //           << "\n";
        update_data.push_back(RepartDistMatrix::pairwise_data{
            linop_id, mode, comm_rank, length, send_id, recv_ptr});
    }
}

template <typename MatrixType>
void generate_alltoall_update_data(
    const ExecutorHandler &exec_handler, std::shared_ptr<SparsityPattern> in,
    std::map<label, scalar *> linops, bool fuse, bool owner,
    label ranks_per_owner,
    std::vector<RepartDistMatrix::all_to_all_data> &update_data)
{
    label linop_offset_store{0};
    for (size_t i = 0; i < 3; i++) {
        label interface_size = in->get_rows()[i].size();
        label linop_idx = (fuse) ? 0 : in->get_id()[i];
        label linop_offset = (fuse) ? linop_offset_store : 0;
        auto comm_pattern = compute_gather_to_owner_counts(
            exec_handler, ranks_per_owner, interface_size);

        size_t recv_size = comm_pattern.recv_offsets.back();

        // NOTE Probably dont need to store linops[linop-idx] because we can
        // just reuse linops
        update_data.emplace_back(
            in->get_id()[i], comm_pattern,
            (owner) ? linops[linop_idx] + linop_offset_store : nullptr);

        linop_offset_store += (fuse) ? recv_size : 0;
    }
}

template <typename MatrixType>
void compute_pad(const std::vector<std::vector<label>> &rows,
                 const std::vector<std::vector<label>> &cols,
                 const std::vector<std::shared_ptr<gko::LinOp>> &linops,
                 std::vector<std::vector<label>> &pads)
{
    for (auto i = 0; i < linops.size(); i++) {
        pads.push_back(std::vector<label>{});
        if constexpr (std::is_same_v<MatrixType,
                                     gko::matrix::Ell<scalar, label>>) {
            auto mtx = gko::as<MatrixType>(linops[i]);
            auto &pad = pads[i];
            label end = mtx->get_num_stored_elements();
            if (end) {
                auto device_exec = mtx->get_executor();
                auto col_dev_view = gko::array<label>::const_view(
                    device_exec, end, mtx->get_const_col_idxs());
                auto ell_cols = col_dev_view.copy_to_array();
                ell_cols.set_executor(device_exec->get_master());


                // sum of all nnzs before
                auto n_rows = (rows[i].size()) ? rows[i].back() + 1 : 0;
                auto nnzs = std::vector<label>(n_rows, 0);
                for (auto j = 0; j < cols[i].size(); j++) {
                    nnzs[rows[i][j]]++;
                }

                auto nnz_sum = std::vector<label>(n_rows + 1, 0);
                for (auto j = 1; j < nnz_sum.size(); j++) {
                    nnz_sum[j] += nnz_sum[j - 1] + nnzs[j - 1];
                }

                // ell inserts values in col major order
                // thus given a linear_index_col_maj we compute the
                // corresponding linear_index_row_maj
                auto coo_pos = std::vector<label>(cols[i].size(), 0);
                auto rel_col = std::vector<label>(end, 0);
                auto el_found = std::vector<label>(n_rows, 0);

                // compute elements found in current row
                for (auto j = 0; j < end; j++) {
                    auto col = ell_cols.get_const_data()[j];
                    if (col >= 0) {
                        // auto row = j %
                        // mtx->get_num_stored_elements_per_row();
                        auto row = j % mtx->get_size()[0];
                        rel_col[j] = el_found[row];
                        el_found[row]++;
                    }
                }

                // iterate in ell order
                auto j_ctr = 0;
                for (auto j = 0; j < end; j++) {
                    auto col = ell_cols.get_const_data()[j];
                    if (col >= 0) {
                        // auto row = j %
                        // mtx->get_num_stored_elements_per_row();
                        auto row = j % mtx->get_size()[0];
                        auto coo_pos_ = nnz_sum[row] + rel_col[j];
                        coo_pos[j_ctr] = coo_pos_;
                        j_ctr++;
                    }
                }

                j_ctr = 0;
                for (auto j = 0; j < end; j++) {
                    auto col = ell_cols.get_const_data()[j];
                    if (col >= 0) {
                        pad.push_back(coo_pos[j_ctr]);
                        j_ctr++;
                    } else {
                        pad.push_back(end - 1);
                    }
                }
            }
        }
    }
}

template <typename MatrixType>
void generate_reorder_map(
    const ExecutorHandler &exec_handler,
    // TODO check if it makes sense to use the linop dataptr map here
    const std::vector<std::shared_ptr<gko::LinOp>> &linops,
    const std::vector<std::vector<label>> &maps,
    const std::vector<std::vector<label>> &pads,
    std::vector<RepartDistMatrix::reorder_map_type> &reorder_maps)
{
    // NOTE early return if rank is empty
    if (maps.size() == 0) return;
    OGL_ASSERT_EQ(linops.size(), maps.size());
    OGL_ASSERT_EQ(linops.size(), pads.size());
    for (size_t i = 0; i < linops.size(); i++) {
        auto &m = maps[i];
        auto &p = pads[i];
        auto map = std::make_shared<gko::array<label>>(
            exec_handler.get_ref_exec(), m.begin(), m.end());
        auto pad = std::make_shared<gko::array<label>>(
            exec_handler.get_ref_exec(), p.begin(), p.end());
        map->set_executor(exec_handler.get_device_exec());
        reorder_maps.emplace_back(
            map, gko::as<MatrixType>(linops[i])->get_values(), pad);
    }
}

template <typename LocalMatrixType>
void RepartDistMatrix::write(const ExecutorHandler &exec_handler,
                             const word field_name, const objectRegistry &db,
                             bool write_global) const
{
    auto local = gko::share(
        gko::matrix::Coo<scalar, label>::create(exec_handler.get_ref_exec()));
    auto non_local = gko::share(
        gko::matrix::Coo<scalar, label>::create(exec_handler.get_ref_exec()));


    if (fuse_) {
        gko::as<LocalMatrixType>(dist_mtx_->get_local_matrix())
            ->convert_to(local.get());
        gko::as<LocalMatrixType>(dist_mtx_->get_non_local_matrix())
            ->convert_to(non_local.get());
    } else {
        gko::as<CombinationMatrix<LocalMatrixType>>(
            dist_mtx_->get_local_matrix())
            ->convert_to(local.get());
        gko::as<CombinationMatrix<LocalMatrixType>>(
            dist_mtx_->get_non_local_matrix())
            ->convert_to(non_local.get());
    }

    if (write_global) {
        size_t rows = dist_mtx_->get_local_matrix()->get_size()[0];
        auto ref_exec = exec_handler.get_ref_exec();
        auto comm = exec_handler.get_host_comm();

        auto partition = gko::share(
            gko::experimental::distributed::build_partition_from_local_size<
                label, label>(ref_exec, *comm.get(), rows));

        // overwrite non_local column indices with global indices
        label rank{exec_handler.get_host_rank()};
        label offset = partition->get_range_bounds()[rank];
        label local_nnz = local->get_num_stored_elements();

        std::transform(local->get_row_idxs(), local->get_row_idxs() + local_nnz,
                       local->get_row_idxs(),
                       [&](label idx) { return idx + offset; });

        std::transform(local->get_col_idxs(), local->get_col_idxs() + local_nnz,
                       local->get_col_idxs(),
                       [&](label idx) { return idx + offset; });

        label non_local_nnz = non_local->get_num_stored_elements();
        std::transform(non_local->get_col_idxs(),
                       non_local->get_col_idxs() + non_local_nnz,
                       non_local->get_col_idxs(), [this](label idx) {
                           return this->compress_to_global()[idx];
                       });

        std::transform(non_local->get_row_idxs(),
                       non_local->get_row_idxs() + non_local_nnz,
                       non_local->get_row_idxs(),
                       [&](label idx) { return idx + offset; });
    }

    export_mtx(field_name + "_local", local, db);
    export_mtx(field_name + "_non_local", non_local, db);
}


/*
 * @map - the corresponding row_major order map view[i] = recv[map[i]]
 * @pad - the corresponding padding map
 */
void reorder_interface_impl(const ExecutorHandler &exec_handler,
                            std::shared_ptr<const gko::array<label>> map,
                            std::shared_ptr<const gko::array<label>> pad,
                            scalar *dst_data)
{
    using vec = gko::matrix::Dense<scalar>;
    using dim_type = gko::dim<2>::dimension_type;

    label recv_size = map->get_size();
    auto device_exec = exec_handler.get_device_exec();

    auto dst_view = gko::array<scalar>::view(device_exec, recv_size, dst_data);

    if (pad->get_num_elems() == 0) {
        // No padding needed
        // a dense view into into dst
        // this allows to row_gather
        auto row_collection = gko::share(gko::matrix::Dense<scalar>::create(
            device_exec, gko::dim<2>{static_cast<dim_type>(recv_size), 1},
            gko::array<scalar>::view(device_exec, recv_size, dst_data), 1));

        auto dense_vec = row_collection->clone();

        dense_vec->row_gather(map.get(), row_collection.get());
    } else {
        // pading after row_gather required, thus data is first row_gathered
        // into a temporary buffer and then padded into final view this is
        // required for example for ELL matrices
        label coo_length = recv_size;
        label ell_length = pad->get_num_elems();
        auto recv_view = gko::share(gko::matrix::Dense<scalar>::create(
            device_exec, gko::dim<2>{static_cast<dim_type>(coo_length), 1},
            gko::array<scalar>::view(device_exec, coo_length, dst_data), 1));
        auto dense_vec = recv_view->clone();

        auto tmp_row_collection = gko::share(gko::matrix::Dense<scalar>::create(
            device_exec, gko::dim<2>{static_cast<dim_type>(ell_length), 1},
            gko::array<scalar>(device_exec, ell_length), 1));
        tmp_row_collection->fill(0.0);
        auto tmp_coo = gko::share(gko::matrix::Dense<scalar>::create(
            device_exec, gko::dim<2>{static_cast<dim_type>(coo_length), 1},
            gko::array<scalar>::view(device_exec, coo_length,
                                     tmp_row_collection->get_values()),
            1));
        dense_vec->row_gather(map.get(), tmp_coo.get());

        // now row gather into final view
        auto row_collection = gko::share(gko::matrix::Dense<scalar>::create(
            device_exec, gko::dim<2>{static_cast<dim_type>(ell_length), 1},
            gko::array<scalar>::view(device_exec, pad->get_num_elems(),
                                     dst_data),
            1));
        tmp_row_collection->row_gather(pad.get(), row_collection.get());
    }
}

void update_impl(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const HostMatrixWrapper> host_A,
    std::vector<RepartDistMatrix::all_to_all_data> &all_to_all_update_data,
    std::vector<RepartDistMatrix::pairwise_data> &pairwise_update_data,
    std::vector<RepartDistMatrix::reorder_map_type> &reorder_maps, bool fuse,
    std::map<label, scalar *> linops, label verbose)
{
    auto comm = exec_handler.get_host_comm();
    auto ref_exec = exec_handler.get_ref_exec();
    auto rank = exec_handler.get_host_rank();
    auto device_exec = exec_handler.get_device_exec();
    bool force_host_buffer = !exec_handler.get_non_orig_device_comm();
    word fieldname = host_A->get_field_name();

    // perform all-to-all updates first
    auto all_to_all_update = [comm, ref_exec, device_exec,
                              all_to_all_update_data, host_A,
                              force_host_buffer]() {
        for (auto [id, comm_pattern, data_ptr] : all_to_all_update_data) {
            auto [length, send_data_ptr] = host_A->get_interface_data(id);
            communicate_values(ref_exec, device_exec, comm, comm_pattern,
                               send_data_ptr, data_ptr, force_host_buffer);
        }
    };

    TIME_WITH_FIELDNAME(verbose, perform_all_to_all_update, fieldname,
                        all_to_all_update(););

    auto get_send_ptr = [&](label send_id, label id) {
        if (send_id < 0) {
            return std::get<1>(host_A->get_interface_data(id));
        } else {
            const scalar *ret = nullptr;
            return ret;
        }
    };


    // perform pairwise communications
    // this update interface data which needs communication
    auto pairwise_communicate = [comm, ref_exec, device_exec,
                                 pairwise_update_data, host_A, rank,
                                 &get_send_ptr, &linops, fuse,
                                 force_host_buffer]() {
        for (auto [id, mode, comm_rank, length, send_id, recv_ptr] :
             pairwise_update_data) {
            std::vector<scalar> send_buffer;
            send_buffer.reserve(length);

            if (mode == 0) {
                const scalar *send_ptr = get_send_ptr(send_id, id);
                for (size_t i = 0; i < length; i++) {
                    send_buffer.push_back(-send_ptr[i]);
                }
                comm->send(ref_exec, send_buffer.data(), length, comm_rank, 0);
            }
            if (mode == 1) {
                if (force_host_buffer) {
                    auto tmp = gko::array<scalar>(ref_exec, length);
                    comm->recv(ref_exec, tmp.get_data(), length, comm_rank, 0);
                    auto recv_view =
                        gko::array<scalar>::view(device_exec, length, recv_ptr);
                    recv_view = tmp;
                } else {
                    comm->recv(device_exec, recv_ptr, length, comm_rank, 0);
                }
            }
            if (mode == 2) {
                const scalar *send_ptr = get_send_ptr(send_id, id);
                for (size_t i = 0; i < length; i++) {
                    send_buffer.push_back(-send_ptr[i]);
                }
                // create views into src and dst
                auto src_view = gko::array<scalar>::const_view(
                    ref_exec, length, send_buffer.data());
                auto dst_view =
                    gko::array<scalar>::view(device_exec, length, recv_ptr);
                dst_view = src_view;
            }
        }
    };

    TIME_WITH_FIELDNAME(verbose, perform_pairwise_update, fieldname,
                        pairwise_communicate(););

    auto reorder_data = [reorder_maps, exec_handler]() {
        for (auto i = 0; i < reorder_maps.size(); i++) {
            auto [reorder_map, data_ptr, pad] = reorder_maps[i];
            // for (auto [reorder_map, data_ptr, pad] : reorder_maps) {
            reorder_interface_impl(exec_handler, reorder_map, pad, data_ptr);
            // }
        }
    };

    TIME_WITH_FIELDNAME(verbose, reorder_matrix_data, fieldname,
                        reorder_data();)
}


template <typename LocalMatrixType>
void RepartDistMatrix::update(const ExecutorHandler &exec_handler,
                              std::shared_ptr<const HostMatrixWrapper> host_A,
                              label verbose)
{
    SIMPLE_TIME(verbose, perform_matrix_update,
                update_impl(exec_handler, host_A, all_to_all_update_data_,
                            pairwise_update_data_, reorder_maps_, fuse_,
                            linops_, verbose););
}


template <typename LocalMatrixType,
          typename NonLocalMatrixType = gko::matrix::Coo<scalar, label>>
std::shared_ptr<RepartDistMatrix> create_impl(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const Repartitioner> repartitioner,
    std::shared_ptr<const HostMatrixWrapper> host_A, word matrix_format,
    bool fuse, label verbose)
{
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;
    label rank = exec_handler.get_host_rank();
    auto exec = exec_handler.get_ref_exec();
    auto host_comm = *exec_handler.get_host_comm().get();
    auto device_comm = *exec_handler.get_device_comm().get();
    bool owner = repartitioner->is_owner(exec_handler);

    auto [local_sparsity, non_local_sparsity] =
        host_A->compute_sparsity_patterns(repartitioner->get_orig_partition());

    auto src_comm_pattern = host_A->create_communication_pattern();
    auto repart_comm_pattern =
        repartitioner->repartition_comm_pattern(exec_handler, src_comm_pattern);

    auto [repart_loc_sparsity, repart_non_loc_sparsity] =
        repartitioner->repartition_sparsity(exec_handler, local_sparsity,
                                            non_local_sparsity);

    auto global_rows = repartitioner->get_orig_partition()->get_size();
    gko::dim<2> global_dim{global_rows, global_rows};
    gko::dim<2> repart_dim{repartitioner->get_repart_size(),
                           repartitioner->get_repart_size()};
    gko::dim<2> repart_non_local_dim{repartitioner->get_repart_size(),
                                     repart_non_loc_sparsity->get_nnz()};

    // create vector of inner type linops
    // if fuse the vector contains only a single element
    // thus we can unwrap it.
    auto device_exec = exec_handler.get_device_exec();
    auto ranks_per_owner = repartitioner->get_ranks_per_gpu();
    bool reparts = ranks_per_owner > 1;
    std::map<label, scalar *> linops;
    auto [loc_rows, loc_cols, loc_map, loc_ids] =
        (fuse) ? repart_loc_sparsity->get_fused_vecs(false)
               : repart_loc_sparsity->get_vecs(false, reparts);
    auto local_linops = generate_inner_linops<LocalMatrixType>(
        exec_handler, repart_dim, loc_rows, loc_cols, loc_ids, linops, fuse);

    auto [non_loc_rows, non_loc_cols, non_loc_map, non_loc_ids] =
        (fuse) ? repart_non_loc_sparsity->get_fused_vecs(true)
               : repart_non_loc_sparsity->get_vecs(true, false);
    auto non_local_linops = generate_inner_linops<NonLocalMatrixType>(
        exec_handler, repart_non_local_dim, non_loc_rows, non_loc_cols,
        non_loc_ids, linops, fuse);

    auto compress_to_global =
        repart_non_loc_sparsity->compute_to_global_map(fuse);

    // compute padding
    std::vector<std::vector<label>> local_pad;
    std::vector<std::vector<label>> non_local_pad;

    // FIXME we pass here loc_rows instead of loc_cols
    // since for symmetric matrices row major loc_rows are column-major cols
    compute_pad<LocalMatrixType>(loc_rows, loc_cols, local_linops, local_pad);
    // TODO pad for non_local is not  needed technically
    compute_pad<NonLocalMatrixType>(non_loc_rows, non_loc_cols,
                                    non_local_linops, non_local_pad);

    // stores original id, comm_patttern, target data ptr
    std::vector<RepartDistMatrix::all_to_all_data> all_to_all_update_data;
    SIMPLE_TIME(verbose, generate_all_to_all_update_data,
                generate_alltoall_update_data<LocalMatrixType>(
                    exec_handler, local_sparsity, linops, fuse, owner,
                    ranks_per_owner, all_to_all_update_data););

    size_t start_local_offset = 0;
    if (fuse && owner) {
        for (size_t i = 0; i < repart_loc_sparsity->get_id().size(); i++) {
            auto id = repart_loc_sparsity->get_id()[i];
            if (id >= 0) {
                start_local_offset += repart_loc_sparsity->get_rows()[i].size();
            }
        }
    }

    std::vector<RepartDistMatrix::pairwise_data> pairwise_update_data;
    SIMPLE_TIME(
        verbose, generate_local_pairwise_data,
        generate_pairwise_update_data<LocalMatrixType>(
            exec_handler, host_A,
            (!owner) ? local_sparsity : repart_loc_sparsity, fuse,
            repartitioner, linops, start_local_offset, pairwise_update_data););
    SIMPLE_TIME(verbose, generate_non_local_pairwise_data,
                generate_pairwise_update_data<NonLocalMatrixType>(
                    exec_handler, host_A,
                    (!owner) ? non_local_sparsity : repart_non_loc_sparsity,
                    fuse, repartitioner, linops, 0, pairwise_update_data););

    // sort the pairwise update_data by interface id so that it is
    // consistent across ranks
    std::stable_sort(pairwise_update_data.begin(), pairwise_update_data.end(),
                     [&](auto &a, auto &b) { return a.id < b.id; });


    std::shared_ptr<dist_mtx> dist_A;
    // recv_gather_idxs are send upon creation to ginkgo distributed matrix
    // to partner ranks. This sets the send_sizes_ on the partner ranks.
    // Thus recv_gather_idxs are local indices of comm partner rank of
    // interfaces.
    auto recv_gather_idxs =
        repart_comm_pattern->compute_recv_gather_idxs(exec_handler);
    auto [send_counts, send_offsets, recv_sizes, recv_offsets] =
        repart_comm_pattern->send_recv_pattern();

    if (verbose > 1) {
        std::ofstream myfile;
        std::string folder = host_A->get_folder();
        myfile.open(folder + "/host_comm_pattern_" +
                    std::to_string(Pstream::myProcNo()));
        myfile << "repart_comm_pattern " << *repart_comm_pattern.get() << "\n";
        myfile << "\nrecv_gather_idxs: " << convert_to_vector(recv_gather_idxs)
               << "\nrecv_sizes: " << recv_sizes
               << "\nrecv_offsets: " << recv_offsets << "\n";
    }

    if (fuse) {
        dist_A = gko::share(dist_mtx::create(
            device_exec, device_comm, global_dim, local_linops[0],
            non_local_linops[0], recv_sizes, recv_offsets, recv_gather_idxs));
    } else {
        dist_A = gko::share(dist_mtx::create(
            device_exec, device_comm, global_dim,
            gko::share(CombinationMatrix<LocalMatrixType>::create(
                device_exec, repart_dim, local_linops)),
            gko::share(CombinationMatrix<NonLocalMatrixType>::create(
                device_exec, repart_non_local_dim, non_local_linops)),
            recv_sizes, recv_offsets, recv_gather_idxs));
    }

    // compute reorder maps
    std::vector<RepartDistMatrix::reorder_map_type> reorder_maps;

    SIMPLE_TIME(
        verbose, generate_local_reorder_map,
        generate_reorder_map<LocalMatrixType>(
            exec_handler, local_linops, loc_map, local_pad, reorder_maps););
    SIMPLE_TIME(verbose, generate_non_local_reorder_map,
                generate_reorder_map<NonLocalMatrixType>(
                    exec_handler, non_local_linops, non_loc_map, non_local_pad,
                    reorder_maps););

    SIMPLE_TIME(verbose, perform_matrix_update,
                update_impl(exec_handler, host_A, all_to_all_update_data,
                            pairwise_update_data, reorder_maps, fuse, linops,
                            verbose););

    return std::make_shared<RepartDistMatrix>(
        device_exec, host_comm, matrix_format, dist_A, repartitioner, fuse,
        all_to_all_update_data, pairwise_update_data, reorder_maps,
        compress_to_global, linops);
}

void write_distributed(const ExecutorHandler &exec_handler, word field_name,
                       const objectRegistry &db,
                       std::shared_ptr<RepartDistMatrix> dist_A,
                       bool write_global)
{
    auto matrix_format{dist_A->get_matrix_format()};
    if (matrix_format == "Coo") {
        return dist_A->write<gko::matrix::Coo<scalar, label>>(
            exec_handler, field_name, db, write_global);
    }
    if (matrix_format == "Csr") {
        return dist_A->write<gko::matrix::Csr<scalar, label>>(
            exec_handler, field_name, db, write_global);
    }
}

void update_distributed(const ExecutorHandler &exec_handler,
                        std::shared_ptr<const HostMatrixWrapper> host_A,
                        std::shared_ptr<RepartDistMatrix> dist_A,
                        word matrix_format, label verbose)
{
    if (matrix_format == "Ell") {
        return dist_A->update<gko::matrix::Ell<scalar, label>>(exec_handler,
                                                               host_A, verbose);
    }
    if (matrix_format == "Coo") {
        return dist_A->update<gko::matrix::Coo<scalar, label>>(exec_handler,
                                                               host_A, verbose);
    }
    if (matrix_format == "Csr") {
        return dist_A->update<gko::matrix::Csr<scalar, label>>(exec_handler,
                                                               host_A, verbose);
    }
}

std::shared_ptr<RepartDistMatrix> create_distributed(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const Repartitioner> repartitioner,
    std::shared_ptr<const HostMatrixWrapper> hostMatrix, word matrix_format,
    bool fuse, label verbose)
{
    if (matrix_format == "Ell") {
        return create_impl<gko::matrix::Ell<scalar, label>>(
            exec_handler, repartitioner, hostMatrix, matrix_format, fuse,
            verbose);
    }
    if (matrix_format == "Coo") {
        return create_impl<gko::matrix::Coo<scalar, label>>(
            exec_handler, repartitioner, hostMatrix, matrix_format, fuse,
            verbose);
    }
    if (matrix_format == "Csr") {
        return create_impl<gko::matrix::Csr<scalar, label>>(
            exec_handler, repartitioner, hostMatrix, matrix_format, fuse,
            verbose);
    }

    FatalErrorInFunction
        << "Matrix format " << matrix_format
        << " not supported. Supported formats are: Ell, Csr, and Coo."
        << abort(FatalError);

    return {};
}
