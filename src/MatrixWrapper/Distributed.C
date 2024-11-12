// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/Distributed.H"

/* helper function to convert a sparsity pattern into a vector of linops with
 * zero coefficients
 *
 */
template <typename MatrixType>
std::vector<std::shared_ptr<gko::LinOp>> generate_inner_linops(
    const ExecutorHandler &exec_handler, gko::dim<2> dim,
    const std::vector<std::vector<label>> &rowss,
    const std::vector<std::vector<label>> &colss)
{
    OGL_ASSERT_EQ(rowss.size(), colss.size());
    auto exec = exec_handler.get_device_exec();
    std::vector<std::shared_ptr<gko::LinOp>> lin_ops;
    // create empty matrix if no interfaces are present
    if (rowss.size() == 0) {
        lin_ops.push_back(
            gko::share(MatrixType::create(exec, gko::dim<2>{0, 0})));
        return lin_ops;
    }
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
        lin_ops.push_back(mtx);
    }
    return lin_ops;
}

template <typename MatrixType>
void generate_update_data(
    const ExecutorHandler &exec_handler, std::shared_ptr<SparsityPattern> in,
    std::vector<std::shared_ptr<gko::LinOp>> &linops, bool fuse,
    label ranks_per_owner,
    std::vector<std::tuple<label, AllToAllPattern, scalar *>> &update_data)
{
    label linop_offset_store{0};
    for (size_t i = 0; i < in->get_id().size(); i++) {
        label interface_size = in->get_rows()[i].size();
        label linop_idx = (fuse) ? 0 : i;
        label linop_offset = (fuse) ? linop_offset_store : 0;
        update_data.emplace_back(
            in->get_id()[i],
            compute_gather_to_owner_counts(exec_handler, ranks_per_owner,
                                           interface_size),
            gko::as<MatrixType>(linops[linop_idx])->get_values() + linop_offset);
        linop_offset_store += interface_size;
    }
}

template <typename MatrixType>
void generate_reorder_map(
    const ExecutorHandler &exec_handler,
    const std::vector<std::shared_ptr<gko::LinOp>> &linops,
    const std::vector<std::vector<label>> &maps,
    std::vector<std::tuple<std::shared_ptr<gko::array<label>>, scalar *>>
        &reorder_maps)
{
    OGL_ASSERT_EQ(linops.size(), maps.size());
    for (size_t i = 0; i < linops.size(); i++) {
        auto &m = maps[i];
        auto map = std::make_shared<gko::array<label>>(
            exec_handler.get_ref_exec(), m.begin(), m.end());
        map->set_executor(exec_handler.get_device_exec());
        reorder_maps.emplace_back(map,
                                  gko::as<MatrixType>(linops[i])->get_values());
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
        FatalErrorInFunction << "Not implemented" << abort(FatalError);
        // overwrite non_local column indices with global indices
        // std::copy(non_local_sparsity_->col_idxs.get_const_data(),
        //           non_local_sparsity_->col_idxs.get_const_data() +
        //               non_local_sparsity_->num_nnz,
        //           non_local->get_col_idxs());

        // auto ref_exec = exec_handler.get_ref_exec();
        // auto comm = exec_handler.get_gko_mpi_host_comm();
        // label rank{exec_handler.get_rank()};
        // auto partition = gko::share(
        //     gko::experimental::distributed::build_partition_from_local_size<
        //         label, label>(ref_exec, *comm.get(),
        //         local_sparsity_->dim[0]));

        // label offset = partition->get_range_bounds()[rank];
        // label local_nnz = local_sparsity_->num_nnz;

        // std::transform(local->get_row_idxs(), local->get_row_idxs() +
        // local_nnz,
        //                local->get_row_idxs(),
        //                [&](label idx) { return idx + offset; });
        // std::transform(local->get_col_idxs(), local->get_col_idxs() +
        // local_nnz,
        //                local->get_col_idxs(),
        //                [&](label idx) { return idx + offset; });

        // label non_local_nnz = non_local_sparsity_->num_nnz;
        // std::transform(non_local->get_row_idxs(),
        //                non_local->get_row_idxs() + non_local_nnz,
        //                non_local->get_row_idxs(),
        //                [&](label idx) { return idx + offset; });
    }

    export_mtx(field_name + "_local", local, db);
    export_mtx(field_name + "_non_local", non_local, db);
}


template <typename LocalMatrixType>
void reorder_interface_impl(const ExecutorHandler &exec_handler,
                            std::shared_ptr<const gko::array<label>>
                                map,  // the corresponding row_major order map
                            scalar *dst_data)
{
    using vec = gko::matrix::Dense<scalar>;
    using dim_type = gko::dim<2>::dimension_type;

    label recv_size = map->get_size();
    auto device_exec = exec_handler.get_device_exec();

    auto dst_view = gko::array<scalar>::view(device_exec, recv_size, dst_data);

    // a dense view into into dst
    // this allows to row_gather
    auto row_collection = gko::share(gko::matrix::Dense<scalar>::create(
        device_exec, gko::dim<2>{static_cast<dim_type>(recv_size), 1},
        gko::array<scalar>::view(device_exec, recv_size, dst_data), 1));

    auto dense_vec = row_collection->clone();
    dense_vec->row_gather(map.get(), row_collection.get());
}

template <typename LocalMatrixType>
void update_impl(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const HostMatrixWrapper> host_A,
    std::vector<std::tuple<label, AllToAllPattern, scalar *>> &update_data,
    std::vector<std::tuple<std::shared_ptr<gko::array<label>>, scalar *>>
        &reorder_maps)
{
    for (auto [id, comm_pattern, data_ptr] : update_data) {
        std::pair<const scalar *, bool> send_data =
            host_A->get_interface_data(id);
        const scalar *send_ptr = std::get<0>(send_data);
        bool neg = std::get<1>(send_data);

        // check if data needs to be inverted
        std::vector<scalar> send_buffer;
        if (neg) {
            auto length = host_A->get_interface_length(id);
            send_buffer.reserve(length);
            for (size_t i = 0; i < length; i++) {
                send_buffer.push_back(send_ptr[i] * -1.0);
            }
        }
        const scalar *send_data_ptr = (neg) ? send_buffer.data() : send_ptr;
        communicate_values(exec_handler, comm_pattern, send_data_ptr, data_ptr);
    }
    for (auto [reorder_map, data_ptr] : reorder_maps) {
        reorder_interface_impl<LocalMatrixType>(exec_handler, reorder_map,
                                                data_ptr);
    }
}


template <typename LocalMatrixType>
void RepartDistMatrix::update(const ExecutorHandler &exec_handler,
                              std::shared_ptr<const HostMatrixWrapper> host_A)
{
    update_impl<LocalMatrixType>(exec_handler, host_A, update_data_,
                                 reorder_maps_);
}


template <typename LocalMatrixType>
std::shared_ptr<RepartDistMatrix> create_impl(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const Repartitioner> repartitioner,
    std::shared_ptr<const HostMatrixWrapper> host_A, word matrix_format,
    bool fuse)
{
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;
    label rank = exec_handler.get_rank();
    auto exec = exec_handler.get_ref_exec();
    auto comm = *exec_handler.get_communicator().get();

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
                                     non_local_sparsity->get_nnz()};

    // create vector of inner type linops
    // if fuse the vector contains only a single element
    // thus we can unwrap it.
    auto device_exec = exec_handler.get_device_exec();
    auto [loc_rows, loc_cols, loc_map] =
        (fuse) ? repart_loc_sparsity->get_fused_vecs(false)
               : repart_loc_sparsity->get_vecs(false);
    auto ranks_per_owner = repartitioner->get_ranks_per_gpu();
    auto local_linops = generate_inner_linops<LocalMatrixType>(
        exec_handler, repart_dim, loc_rows, loc_cols);

    auto [non_loc_rows, non_loc_cols, non_loc_map] =
        (fuse) ? repart_non_loc_sparsity->get_fused_vecs(true)
               : repart_non_loc_sparsity->get_vecs(true);

    auto non_local_linops = generate_inner_linops<LocalMatrixType>(
        exec_handler, repart_non_local_dim, non_loc_rows, non_loc_cols);

    std::vector<std::tuple<label, AllToAllPattern, scalar *>> update_data;
    generate_update_data<LocalMatrixType>(exec_handler, repart_loc_sparsity,
                                          local_linops, fuse, ranks_per_owner,
                                          update_data);
    generate_update_data<LocalMatrixType>(exec_handler, repart_non_loc_sparsity,
                                          non_local_linops, fuse,
                                          ranks_per_owner, update_data);

    std::shared_ptr<dist_mtx> dist_A;
    // recv_gather_idxs are send upon creation to ginkgo distributed matrix
    // to partner ranks. This sets the send_sizes_ on the partner ranks.
    // Thus recv_gather_idxs are local indices of comm partner rank of
    // interfaces.
    auto recv_gather_idxs =
        repart_comm_pattern->compute_recv_gather_idxs(exec_handler);
    auto [send_counts, send_offsets, recv_sizes, recv_offsets] =
        repart_comm_pattern->send_recv_pattern();

    if (fuse) {
        dist_A = gko::share(dist_mtx::create(
            device_exec, comm, global_dim, local_linops[0], non_local_linops[0],
            recv_sizes, recv_offsets, recv_gather_idxs));
    } else {
        dist_A = gko::share(dist_mtx::create(
            device_exec, comm, global_dim,
            gko::share(CombinationMatrix<LocalMatrixType>::create(
                device_exec, repart_dim, local_linops)),
            gko::share(CombinationMatrix<LocalMatrixType>::create(
                device_exec, repart_non_local_dim, non_local_linops)),
            recv_sizes, recv_offsets, recv_gather_idxs));
    }

    // compute reorder maps
    std::vector<std::tuple<std::shared_ptr<gko::array<label>>, scalar *>>
        reorder_maps;
    generate_reorder_map<LocalMatrixType>(exec_handler, local_linops, loc_map,
                                          reorder_maps);
    generate_reorder_map<LocalMatrixType>(exec_handler, non_local_linops,
                                          non_loc_map, reorder_maps);

    update_impl<LocalMatrixType>(exec_handler, host_A, update_data,
                                 reorder_maps);

    return std::make_shared<RepartDistMatrix>(
        device_exec, comm, matrix_format, dist_A, src_comm_pattern,
        repart_comm_pattern, repartitioner, fuse, update_data, reorder_maps);
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
                        word matrix_format)
{
    if (matrix_format == "Ell") {
        FatalErrorInFunction
            << " Updating Ell matrix not supported\nSet regenerate 1;"
            << exit(FatalError);
    }
    if (matrix_format == "Coo") {
        return dist_A->update<gko::matrix::Coo<scalar, label>>(exec_handler,
                                                               host_A);
    }
    if (matrix_format == "Csr") {
        return dist_A->update<gko::matrix::Csr<scalar, label>>(exec_handler,
                                                               host_A);
    }
}

std::shared_ptr<RepartDistMatrix> create_distributed(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const Repartitioner> repartitioner,
    std::shared_ptr<const HostMatrixWrapper> hostMatrix, word matrix_format,
    bool fuse)
{
    if (matrix_format == "Ell") {
        return create_impl<gko::matrix::Ell<scalar, label>>(
            exec_handler, repartitioner, hostMatrix, matrix_format, fuse);
    }
    if (matrix_format == "Coo") {
        return create_impl<gko::matrix::Coo<scalar, label>>(
            exec_handler, repartitioner, hostMatrix, matrix_format, fuse);
    }
    if (matrix_format == "Csr") {
        return create_impl<gko::matrix::Csr<scalar, label>>(
            exec_handler, repartitioner, hostMatrix, matrix_format, fuse);
    }

    FatalErrorInFunction
        << "Matrix format " << matrix_format
        << " not supported. Supported formats are: Ell, Csr, and Coo."
        << abort(FatalError);

    return {};
}
