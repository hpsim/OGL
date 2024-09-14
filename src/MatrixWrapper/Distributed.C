// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/Distributed.H"

template <typename MatrixType>
std::vector<std::shared_ptr<gko::LinOp>> generate_inner_linops(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const SparsityPattern> sparsity, bool compress_cols)
{
    std::vector<std::shared_ptr<gko::LinOp>> lin_ops;
    auto compress_columns = [](std::vector<label> &in) {
        std::vector<label> tmp = in;
        std::stable_sort(tmp.begin(), tmp.end());

        std::map<label, label> key_map;
        label ctr{0};
        for (auto el : tmp) {
            key_map[el] = ctr;
            ctr++;
        }

        for (size_t i = 0; i < in.size(); i++) {
            in[i] = key_map[in[i]];
        }
    };

    std::vector<label> tmp_cols(
        sparsity->col_idxs.get_const_data(),
        sparsity->col_idxs.get_const_data() + sparsity->num_nnz);
    if (compress_cols) {
        compress_columns(tmp_cols);
    }
    if (sparsity->spans.size() == 0) {
        lin_ops.push_back(
            gko::share(MatrixType::create(exec, gko::dim<2>{0, 0})));
        return lin_ops;
    }
    for (size_t i = 0; i < sparsity->spans.size(); i++) {
        auto [begin, end] = sparsity->spans[i];
        gko::array<scalar> coeffs(exec, end - begin);
        coeffs.fill(0.0);
        auto mtx_data = gko::device_matrix_data<scalar, label>(
            exec->get_master(), sparsity->dim,
            gko::array<label>(exec->get_master(),
                              sparsity->row_idxs.get_const_data() + begin,
                              sparsity->row_idxs.get_const_data() + end),
            gko::array<label>(exec->get_master(), tmp_cols.data() + begin,
                              tmp_cols.data() + end),
            coeffs);
        auto mtx = gko::share(MatrixType::create(exec));
        gko::as<MatrixType>(mtx)->read(mtx_data);
        lin_ops.push_back(mtx);
    }
    return lin_ops;
}


template <typename LocalMatrixType>
void RepartDistMatrix::write(const ExecutorHandler &exec_handler,
                             const word field_name,
                             const objectRegistry &db) const
{
    if (repartitioner_->get_fused()) {
        auto ret_local =
            gko::as<LocalMatrixType>(dist_mtx_->get_local_matrix());
        auto coo_local = gko::share(gko::matrix::Coo<scalar, label>::create(
            exec_handler.get_ref_exec()));
        ret_local->convert_to(coo_local.get());
        export_mtx(field_name + "_local", coo_local, db);

        auto ret_non_local =
            gko::as<LocalMatrixType>(dist_mtx_->get_non_local_matrix());
        auto coo_non_local = gko::share(gko::matrix::Coo<scalar, label>::create(
            exec_handler.get_ref_exec()));
        ret_non_local->convert_to(coo_non_local.get());
        export_mtx(field_name + "_non_local", coo_non_local, db);
    } else {
        auto ret = gko::share(gko::matrix::Coo<scalar, label>::create(
            exec_handler.get_ref_exec()));
        gko::as<CombinationMatrix<LocalMatrixType>>(
            dist_mtx_->get_local_matrix())
            ->convert_to(ret.get());
        export_mtx(field_name + "_local", ret, db);

        auto non_loc_ret = gko::share(gko::matrix::Coo<scalar, label>::create(
            exec_handler.get_ref_exec()));
        gko::as<CombinationMatrix<LocalMatrixType>>(
            dist_mtx_->get_non_local_matrix())
            ->convert_to(non_loc_ret.get());

        // overwrite with global
        bool write_global = true;
        if (write_global) {
            std::copy(non_local_sparsity_->col_idxs.get_const_data(),
                      non_local_sparsity_->col_idxs.get_const_data() +
                          non_local_sparsity_->num_nnz,
                      non_loc_ret->get_col_idxs());
        }
        export_mtx(field_name + "_non_local", non_loc_ret, db);
    }
}

template <typename LocalMatrixType>
void update_fused_impl(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const Repartitioner> repartitioner,
    std::shared_ptr<const HostMatrixWrapper> host_A,
    std::shared_ptr<
        gko::experimental::distributed::Matrix<scalar, label, label>>
        dist_A,
    std::shared_ptr<const SparsityPattern> local_sparsity,
    std::shared_ptr<const SparsityPattern> non_local_sparsity,
    [[maybe_unused]] std::shared_ptr<const CommunicationPattern> src_comm_pattern,
    std::vector<std::tuple<bool, label, label>> local_interfaces)
{
    auto exec = exec_handler.get_ref_exec();
    auto device_exec = exec_handler.get_device_exec();
    auto ranks_per_gpu = repartitioner->get_ranks_per_gpu();
    [[maybe_unused]] bool requires_host_buffer =
        exec_handler.get_gko_force_host_buffer();

    label rank{exec_handler.get_rank()};
    // label owner_rank = repartitioner->get_owner_rank(exec_handler);
    bool owner = repartitioner->is_owner(exec_handler);
    label nrows = host_A->get_local_nrows();
    label local_matrix_nnz = host_A->get_local_matrix_nnz();
    label n_interfaces = 0;  // number of fused interface coefficients
    for (size_t i = 0; i < local_interfaces.size(); i++) {
        auto [local, rank, size] = local_interfaces[i];
        if (local) {
            n_interfaces += size;
        }
    }
    label tot_local_matrix_nnz = local_matrix_nnz + n_interfaces;

    // size + padding has to be local_matrix_nnz
    // [upper, lower, diag, interfaces]
    auto diag_comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, nrows, tot_local_matrix_nnz,
        local_matrix_nnz - nrows, n_interfaces);
    label upper_nnz = host_A->get_upper_nnz();
    auto upper_comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, upper_nnz, tot_local_matrix_nnz, 0,
        tot_local_matrix_nnz - upper_nnz);
    auto lower_comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, upper_nnz, tot_local_matrix_nnz, upper_nnz,
        nrows + n_interfaces);

    // label nnz = 0;
    //
    // update main values
    std::vector<scalar> loc_buffer;
    auto local_mtx = gko::as<LocalMatrixType>(dist_A->get_local_matrix());
    scalar *local_ptr =
        (owner) ? const_cast<scalar *>(local_mtx->get_const_values()) : nullptr;

    communicate_values(exec_handler, diag_comm_pattern, host_A->get_diag(),
                       local_ptr);
    communicate_values(exec_handler, upper_comm_pattern, host_A->get_upper(),
                       local_ptr);

    if (host_A->get_symmetric()) {
        // TODO FIXME
        // if symmetric we can reuse already copied data
        communicate_values(exec_handler, lower_comm_pattern,
                           host_A->get_lower(), local_ptr);
    } else {
        communicate_values(exec_handler, lower_comm_pattern,
                           host_A->get_lower(), local_ptr);
    }

    // copy interface values
    auto comm = *exec_handler.get_communicator().get();
    if (owner) {
        std::shared_ptr<const LocalMatrixType> mtx;
        label loc_ctr{1};
        label nloc_ctr{0};
        label host_interface_ctr{0};
        // label tag = 0;
        label begin = 0;
        scalar *recv_buffer_ptr;
        for (auto [is_local, orig_rank, size] : local_interfaces) {
            label &ctr = (is_local) ? loc_ctr : nloc_ctr;
            if (is_local) {
                // TODO if fused it is only operator 0
                mtx =
                    gko::as<const LocalMatrixType>(dist_A->get_local_matrix());
            } else {
                mtx = gko::as<const LocalMatrixType>(
                    dist_A->get_non_local_matrix());
            }
            ctr++;

            recv_buffer_ptr = const_cast<scalar *>(mtx->get_const_values());

            if (orig_rank == rank) {
                // if data is already on this rank
                // TODO probably better if we handle this case separately
                auto data_view = gko::array<scalar>::const_view(
                    exec, size, host_A->get_interface_data(host_interface_ctr));

                auto target_view = gko::array<scalar>::view(
                    mtx->get_executor(), size, recv_buffer_ptr + begin);
                begin += size;

                target_view = data_view;
                host_interface_ctr++;
            } else {
                // data is not already on rank
                // comm.recv(device_exec, recv_buffer_ptr, size, orig_rank,
                // tag);
            }
        }
        // interface values need to be multiplied by -1
        using vec = gko::matrix::Dense<scalar>;
        mtx = gko::as<const LocalMatrixType>(dist_A->get_non_local_matrix());
        recv_buffer_ptr = const_cast<scalar *>(mtx->get_const_values());
        auto neg_one = gko::initialize<vec>({-1.0}, device_exec);
        auto interface_dense = vec::create(
            mtx->get_executor(),
            gko::dim<2>{
                static_cast<gko::size_type>(mtx->get_num_stored_elements()), 1},
            gko::array<scalar>::view(mtx->get_executor(),
                                     mtx->get_num_stored_elements(),
                                     recv_buffer_ptr),
            1);
        interface_dense->scale(neg_one);
    } else {
        // the non owner has send all its interfaces to owner
        // thus all values need to be communicated to the owner as well
        // label num_interfaces = src_comm_pattern->target_ids.size();
        // label tag = 0;
        // for (int i = 0; i < num_interfaces; i++) {
        //     label comm_size = src_comm_pattern->target_sizes.data()[i];
        //     const scalar *send_buffer_ptr = host_A->get_interface_data(i);
        //     comm.send(device_exec, send_buffer_ptr, comm_size, owner_rank,
        //     tag);
        // }
    }

    // reorder updated values
    if (owner) {
        // local data
        using dim_type = gko::dim<2>::dimension_type;
        auto local_elements = local_mtx->get_num_stored_elements();

        // TODO make sure this doesn't copy
        // create a non owning dense matrix of local_values
        auto row_collection = gko::share(gko::matrix::Dense<scalar>::create(
            local_mtx->get_executor(),
            gko::dim<2>{static_cast<dim_type>(local_elements), 1},
            gko::array<scalar>::view(local_mtx->get_executor(), local_elements,
                                     local_ptr),
            1));

        local_sparsity->ldu_mapping.set_executor(dist_A->get_executor());
        auto mapping_view = gko::array<label>::view(
            local_sparsity->ldu_mapping.get_executor(), local_elements,
            local_sparsity->ldu_mapping.get_data());

        auto dense_vec = row_collection->clone();
        dense_vec->row_gather(&mapping_view, row_collection.get());


        // non local data
        auto non_local_mtx = gko::as<LocalMatrixType>(dist_A->get_non_local_matrix());
        auto non_local_elements = non_local_mtx->get_num_stored_elements();
    scalar *non_local_ptr =
        (owner) ? const_cast<scalar *>(non_local_mtx->get_const_values()) : nullptr;

        // TODO make sure this doesn't copy
        // create a non owning dense matrix of local_values
        auto non_local_row_collection = gko::share(gko::matrix::Dense<scalar>::create(
            non_local_mtx->get_executor(),
            gko::dim<2>{static_cast<dim_type>(non_local_elements), 1},
            gko::array<scalar>::view(non_local_mtx->get_executor(), non_local_elements,
                                     non_local_ptr),
            1));

        non_local_sparsity->ldu_mapping.set_executor(dist_A->get_executor());
        auto non_local_mapping_view = gko::array<label>::view(
           non_local_sparsity->ldu_mapping.get_executor(), non_local_elements,
           non_local_sparsity->ldu_mapping.get_data());

        auto non_local_dense_vec = non_local_row_collection->clone();
        non_local_dense_vec->row_gather(&non_local_mapping_view, non_local_row_collection.get());
    }
}

template <typename LocalMatrixType>
void update_impl(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const Repartitioner> repartitioner,
    std::shared_ptr<const HostMatrixWrapper> host_A,
    std::shared_ptr<
        gko::experimental::distributed::Matrix<scalar, label, label>>
        dist_A,
    std::shared_ptr<const SparsityPattern> local_sparsity,
    [[maybe_unused]] std::shared_ptr<const SparsityPattern> non_local_sparsity,
    std::shared_ptr<const CommunicationPattern> src_comm_pattern,
    std::vector<std::tuple<bool, label, label>> local_interfaces)
{
    auto exec = exec_handler.get_ref_exec();
    auto device_exec = exec_handler.get_device_exec();
    auto ranks_per_gpu = repartitioner->get_ranks_per_gpu();
    [[maybe_unused]] bool requires_host_buffer =
        exec_handler.get_gko_force_host_buffer();

    label rank{exec_handler.get_rank()};
    label owner_rank = repartitioner->get_owner_rank(exec_handler);
    bool owner = repartitioner->is_owner(exec_handler);
    label nrows = host_A->get_local_nrows();
    label local_matrix_nnz = host_A->get_local_matrix_nnz();

    // size + padding has to be local_matrix_nnz
    auto diag_comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, nrows, local_matrix_nnz,
        local_matrix_nnz - nrows, 0);
    label upper_nnz = host_A->get_upper_nnz();
    auto upper_comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, upper_nnz, local_matrix_nnz, 0,
        local_matrix_nnz - upper_nnz);
    auto lower_comm_pattern =
        compute_gather_to_owner_counts(exec_handler, ranks_per_gpu, upper_nnz,
                                       local_matrix_nnz, upper_nnz, nrows);

    scalar *local_ptr = nullptr;
    // label nnz = 0;

    // update main values
    std::vector<scalar> loc_buffer;
    auto local_mtx =
        gko::as<CombinationMatrix<LocalMatrixType>>(dist_A->get_local_matrix());

    std::shared_ptr<const LocalMatrixType> local =
        (owner) ? gko::as<LocalMatrixType>(local_mtx->get_operators()[0])
                : std::shared_ptr<const LocalMatrixType>{};

    if (owner) {
        local_ptr = const_cast<scalar *>(local->get_const_values());
    }

    communicate_values(exec_handler, diag_comm_pattern, host_A->get_diag(),
                       local_ptr);
    communicate_values(exec_handler, upper_comm_pattern, host_A->get_upper(),
                       local_ptr);

    if (host_A->get_symmetric()) {
        // TODO FIXME
        // if symmetric we can reuse already copied data
        communicate_values(exec_handler, lower_comm_pattern,
                           host_A->get_lower(), local_ptr);
    } else {
        communicate_values(exec_handler, lower_comm_pattern,
                           host_A->get_lower(), local_ptr);
    }

    // copy interface values
    auto comm = *exec_handler.get_communicator().get();
    if (owner) {
        std::shared_ptr<const LocalMatrixType> mtx;
        label loc_ctr{1};
        label nloc_ctr{0};
        label host_interface_ctr{0};
        label tag = 0;
        scalar *recv_buffer_ptr;
        // scalar *recv_buffer_ptr_2;
        // std::vector<scalar> host_recv_buffer;
        label remain_host_interfaces = host_A->get_num_interfaces();
        for (auto [is_local, orig_rank, size] : local_interfaces) {
            label &ctr = (is_local) ? loc_ctr : nloc_ctr;
            if (is_local) {
                // TODO if fused it is only operator 0
                mtx = gko::as<LocalMatrixType>(
                    gko::as<CombinationMatrix<LocalMatrixType>>(
                        dist_A->get_local_matrix())
                        ->get_combination()
                        ->get_operators()[ctr]);
            } else {
                mtx = gko::as<LocalMatrixType>(
                    gko::as<CombinationMatrix<LocalMatrixType>>(
                        dist_A->get_non_local_matrix())
                        ->get_combination()
                        ->get_operators()[ctr]);
            }
            ctr++;

            // FIXME TODO needs to offset by interface start for fusing
            recv_buffer_ptr = const_cast<scalar *>(mtx->get_const_values());
            // }

            if (orig_rank != rank) {
                // data is not already on rank
                comm.recv(device_exec, recv_buffer_ptr, size, orig_rank, tag);
            } else {
                // if data is already on this rank
                // TODO probably better if we handle this case separately
                auto data_view = gko::array<scalar>::const_view(
                    exec, size, host_A->get_interface_data(host_interface_ctr));

                auto target_view = gko::array<scalar>::view(
                    mtx->get_executor(), size, recv_buffer_ptr);

                target_view = data_view;

                host_interface_ctr++;
                remain_host_interfaces--;
            }

            // interface values need to be multiplied by -1
            using vec = gko::matrix::Dense<scalar>;
            recv_buffer_ptr = const_cast<scalar *>(mtx->get_const_values());
            auto neg_one = gko::initialize<vec>({-1.0}, device_exec);
            auto interface_dense =
                vec::create(local->get_executor(),
                            gko::dim<2>{static_cast<gko::size_type>(size), 1},
                            gko::array<scalar>::view(local->get_executor(),
                                                     size, recv_buffer_ptr),
                            1);
            interface_dense->scale(neg_one);
        }
    } else {
        // the non owner has send all its interfaces to owner
        // thus all values need to be communicated to the owner as well
        label num_interfaces = src_comm_pattern->target_ids.size();
        label tag = 0;
        for (int i = 0; i < num_interfaces; i++) {
            label size = src_comm_pattern->target_sizes.data()[i];
            const scalar *send_buffer_ptr = host_A->get_interface_data(i);
            comm.send(device_exec, send_buffer_ptr, size, owner_rank, tag);
        }
    }

    // reorder updated values
    if (owner) {
        // TODO reorder everything doesnt need row_collection clone
        using dim_type = gko::dim<2>::dimension_type;

        std::shared_ptr<const LocalMatrixType> local =
        gko::as<LocalMatrixType>(
            gko::as<CombinationMatrix<LocalMatrixType>>(
                dist_A->get_local_matrix())
                ->get_combination()
                ->get_operators()[0]);

        auto local_elements = local->get_num_stored_elements();
        local_ptr = const_cast<scalar *>(local->get_const_values());

        // TODO make sure this doesn't copy
        // create a non owning dense matrix of local_values
        auto row_collection = gko::share(gko::matrix::Dense<scalar>::create(
            local->get_executor(),
            gko::dim<2>{static_cast<dim_type>(local_elements), 1},
            gko::array<scalar>::view(local->get_executor(), local_elements,
                                     local_ptr),
            1));

        local_sparsity->ldu_mapping.set_executor(dist_A->get_executor());
        auto mapping_view = gko::array<label>::view(
            local_sparsity->ldu_mapping.get_executor(), local_elements,
            local_sparsity->ldu_mapping.get_data());

        auto dense_vec = row_collection->clone();
        dense_vec->row_gather(&mapping_view, row_collection.get());
    }
}


template <typename InnerType>
void RepartDistMatrix::update(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const Repartitioner> repartitioner,
    std::shared_ptr<const HostMatrixWrapper> host_A)
{

    if (repartitioner->get_fused()) {
    update_fused_impl<InnerType>(exec_handler, repartitioner, host_A, dist_mtx_,
                           local_sparsity_, non_local_sparsity_,
                           src_comm_pattern_, local_interfaces_);
    } else {
    update_impl<InnerType>(exec_handler, repartitioner, host_A, dist_mtx_,
                           local_sparsity_, non_local_sparsity_,
                           src_comm_pattern_, local_interfaces_);

    }
}


template <typename LocalMatrixType>
std::shared_ptr<RepartDistMatrix> create_impl(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const Repartitioner> repartitioner,
    std::shared_ptr<const HostMatrixWrapper> host_A)
{
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;
    label rank = exec_handler.get_rank();
    auto exec = exec_handler.get_ref_exec();
    auto comm = *exec_handler.get_communicator().get();

    auto local_sparsity = host_A->compute_local_sparsity(exec);
    auto non_local_sparsity = host_A->compute_non_local_sparsity(exec);

    auto src_comm_pattern = host_A->create_communication_pattern();
    if (non_local_sparsity->spans.size() != src_comm_pattern->target_ids.size()) {
        FatalErrorInFunction << " Inconsistency detected non_local_sparsity->spans.size() !=src_comm_pattern->target_ids.size() on rank" << exit(FatalError);
    }

    auto repart_comm_pattern =
        repartitioner->repartition_comm_pattern(exec_handler, src_comm_pattern);

    auto tmp_send_global_cols = detail::convert_to_global(
        repartitioner->get_orig_partition(),
        non_local_sparsity->col_idxs.get_const_data(),
        non_local_sparsity->spans, src_comm_pattern->target_ids);

    std::copy(tmp_send_global_cols.begin(), tmp_send_global_cols.end(),
              non_local_sparsity->col_idxs.get_data());

    auto [repart_loc_sparsity, repart_non_loc_sparsity, local_interfaces] =
        repartitioner->repartition_sparsity(
            exec_handler, local_sparsity, non_local_sparsity,
            src_comm_pattern->target_ids, repartitioner->get_fused());

    // create vector of inner type linops
    // if fuse the vector contains only a single element
    // thus we can unwrap it.
    auto device_exec = exec_handler.get_device_exec();
    auto local_linops = generate_inner_linops<LocalMatrixType>(
        device_exec, repart_loc_sparsity, false);
    auto non_local_linops = generate_inner_linops<LocalMatrixType>(
        device_exec, repart_non_loc_sparsity, true);

    auto [send_counts, send_offsets, recv_sizes, recv_offsets] =
        repart_comm_pattern->send_recv_pattern();

    // recv_gather_idxs are send upon creation to ginkgo distributed matrix
    // to partner ranks. This sets the send_sizes_ on the partner ranks.
    // Thus recv_gather_idxs are local indices of comm partner rank of
    // interfaces.
    auto recv_gather_idxs =
        repart_comm_pattern->compute_recv_gather_idxs(exec_handler);


    std::cout << __FILE__ << " rank " << rank
        << "recv_gather " <<  convert_to_vector(recv_gather_idxs)
        << " recv_sizes " <<  recv_sizes
        << " recv_offsets " <<  recv_offsets
        << "\n";

    auto global_rows = repartitioner->get_orig_partition()->get_size();
    gko::dim<2> global_dim{global_rows, global_rows};

    std::shared_ptr<dist_mtx> dist_A;
    if (repartitioner->get_fused()) {
        dist_A = gko::share(dist_mtx::create(
            device_exec, comm, global_dim, local_linops[0], non_local_linops[0],
            recv_sizes, recv_offsets, recv_gather_idxs));
        update_fused_impl<LocalMatrixType>(
            exec_handler, repartitioner, host_A, dist_A, repart_loc_sparsity,
            repart_non_loc_sparsity, src_comm_pattern, local_interfaces);
    } else {
        dist_A = gko::share(dist_mtx::create(
            device_exec, comm, global_dim,
            gko::share(CombinationMatrix<LocalMatrixType>::create(
                device_exec, repart_loc_sparsity->dim, local_linops)),
            gko::share(CombinationMatrix<LocalMatrixType>::create(
                device_exec, repart_non_loc_sparsity->dim, non_local_linops)),
            recv_sizes, recv_offsets, recv_gather_idxs));
        update_impl<LocalMatrixType>(
            exec_handler, repartitioner, host_A, dist_A, repart_loc_sparsity,
            repart_non_loc_sparsity, src_comm_pattern, local_interfaces);
    }

    return std::make_shared<RepartDistMatrix>(
        device_exec, comm, dist_A, repart_loc_sparsity, repart_non_loc_sparsity,
        src_comm_pattern, repart_comm_pattern, repartitioner, local_interfaces);
}

std::shared_ptr<const gko::LinOp> get_local(std::shared_ptr<const gko::LinOp> dist_A
                       )
{
    word matrix_format = "Coo";
    if (matrix_format == "Coo") {
        return gko::as<RepartDistMatrix>(dist_A)->get_local<const gko::matrix::Coo<scalar, label>>();
    }
    if (matrix_format == "Csr") {
        return gko::as<RepartDistMatrix>(dist_A)->get_local<const gko::matrix::Csr<scalar, label>>();
    }

}

void write_distributed(const ExecutorHandler &exec_handler, word field_name,
                       const objectRegistry &db,
                       std::shared_ptr<RepartDistMatrix> dist_A,
                       word matrix_format)
{
    if (matrix_format == "Coo") {
        return dist_A->write<gko::matrix::Coo<scalar, label>>(exec_handler,
                                                              field_name, db);
    }
    if (matrix_format == "Csr") {
        return dist_A->write<gko::matrix::Csr<scalar, label>>(exec_handler,
                                                              field_name, db);
    }
}

void update_distributed(const ExecutorHandler &exec_handler,
                        std::shared_ptr<const Repartitioner> repartitioner,
                        std::shared_ptr<const HostMatrixWrapper> host_A,
                        std::shared_ptr<RepartDistMatrix> dist_A,
                        word matrix_format)
{
    if (matrix_format == "Ell") {
        FatalErrorInFunction
            << " Updating Ell matrix not supported\nSet regenerate 1;"
            << exit(FatalError);
    }
    if (matrix_format == "Coo") {
        return dist_A->update<gko::matrix::Coo<scalar, label>>(
            exec_handler, repartitioner, host_A);
    }
    if (matrix_format == "Csr") {
        return dist_A->update<gko::matrix::Csr<scalar, label>>(
            exec_handler, repartitioner, host_A);
    }
}

std::shared_ptr<RepartDistMatrix> create_distributed(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const Repartitioner> repartitioner,
    std::shared_ptr<const HostMatrixWrapper> hostMatrix, word matrix_format)
{
    if (matrix_format == "Ell") {
        return create_impl<gko::matrix::Ell<scalar, label>>(
            exec_handler, repartitioner, hostMatrix);
    }
    if (matrix_format == "Coo") {
        return create_impl<gko::matrix::Coo<scalar, label>>(
            exec_handler, repartitioner, hostMatrix);
    }
    if (matrix_format == "Csr") {
        return create_impl<gko::matrix::Csr<scalar, label>>(
            exec_handler, repartitioner, hostMatrix);
    }

    FatalErrorInFunction
        << "Matrix format " << matrix_format
        << " not supported. Supported formats are: Ell, Csr, and Coo."
        << abort(FatalError);

    return {};
}
