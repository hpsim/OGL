// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/Distributed.H"

std::vector<std::shared_ptr<const gko::LinOp>> detail::generate_inner_linops(
    word matrix_format, bool fuse, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const SparsityPattern> sparsity)
{
    std::vector<std::shared_ptr<const gko::LinOp>> lin_ops;
    for (int i = 0; i < sparsity->spans.size(); i++) {
        auto [begin, end] = sparsity->spans[i];
        gko::array<scalar> coeffs(exec, end - begin);
        coeffs.fill(0.0);

        auto row_idxs_arr = gko::array<label>(
            exec->get_master(), sparsity->row_idxs.get_const_data() + begin,
            sparsity->row_idxs.get_const_data() + end);
        auto col_idxs_arr = gko::array<label>(
            exec->get_master(), sparsity->col_idxs.get_const_data() + begin,
            sparsity->col_idxs.get_const_data() + end);

        auto create_function = [&](word format) {
            return gko::share(gko::matrix::Coo<scalar, label>::create(
                exec, sparsity->dim, coeffs, col_idxs_arr, row_idxs_arr));
        };
        lin_ops.push_back(create_function(matrix_format));
    }
    return lin_ops;
}

void detail::update_impl(
    const ExecutorHandler &exec_handler, word matrix_format,
    const Repartitioner &repartitioner,
    std::shared_ptr<const HostMatrixWrapper> host_A,
    std::shared_ptr<
        gko::experimental::distributed::Matrix<scalar, label, label>>
        dist_A,
    std::shared_ptr<const SparsityPattern> local_sparsity,
    std::shared_ptr<const SparsityPattern> non_local_sparsity,
    std::shared_ptr<const CommunicationPattern> src_comm_pattern,
    std::vector<std::pair<bool, label>> local_interfaces)
{
    auto exec = exec_handler.get_ref_exec();
    auto device_exec = exec_handler.get_device_exec();
    auto ranks_per_gpu = repartitioner.get_ranks_per_gpu();
    bool requires_host_buffer = exec_handler.get_gko_force_host_buffer();

    label rank{exec_handler.get_rank()};
    label owner_rank = repartitioner.get_owner_rank(exec_handler);
    bool owner = repartitioner.is_owner(exec_handler);
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

    scalar *local_ptr;
    scalar *local_ptr_2;
    label nnz = 0;

    // update main values
    std::vector<scalar> loc_buffer;
    if (owner) {
        using Coo = gko::matrix::Coo<scalar, label>;

        // TODO make it work with other matrix types
        std::shared_ptr<const Coo> local = gko::as<Coo>(
            gko::as<CombinationMatrix<Coo>>(dist_A->get_local_matrix())
                ->get_combination()
                ->get_operators()[0]);

        nnz = local->get_num_stored_elements();
        if (requires_host_buffer) {
            loc_buffer.resize(nnz);
            local_ptr = loc_buffer.data();
            local_ptr_2 = const_cast<scalar *>(local->get_const_values());
        } else {
            local_ptr = const_cast<scalar *>(local->get_const_values());
        }
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

    if (requires_host_buffer) {
        auto host_buffer_view = gko::array<scalar>::view(exec, nnz, local_ptr);
        auto target_buffer_view =
            gko::array<scalar>::view(device_exec, nnz, local_ptr_2);
        target_buffer_view = host_buffer_view;
    }

    // copy interface values
    auto comm = *exec_handler.get_communicator().get();
    if (owner) {
        using Coo = gko::matrix::Coo<scalar, label>;
        std::shared_ptr<const Coo> mtx;
        label loc_ctr{1};
        label nloc_ctr{0};
        label host_interface_ctr{0};
        label tag = 0;
        label comm_rank, comm_size;
        scalar *recv_buffer_ptr;
        scalar *recv_buffer_ptr_2;
        std::vector<scalar> host_recv_buffer;
        label remain_host_interfaces = host_A->get_num_interfaces();
        for (auto [is_local, comm_rank] : local_interfaces) {
            label &ctr = (is_local) ? loc_ctr : nloc_ctr;
            if (is_local) {
                // TODO make it work for other types
                mtx = gko::as<Coo>(
                    gko::as<CombinationMatrix<Coo>>(dist_A->get_local_matrix())
                        ->get_combination()
                        ->get_operators()[ctr]);
                comm_size = local_sparsity->spans[ctr].length();
            } else {
                mtx = gko::as<Coo>(gko::as<CombinationMatrix<Coo>>(
                                       dist_A->get_non_local_matrix())
                                       ->get_combination()
                                       ->get_operators()[ctr]);
                comm_size = non_local_sparsity->spans[ctr].length();
            }

            if (requires_host_buffer) {
                host_recv_buffer.resize(comm_size);
                recv_buffer_ptr = host_recv_buffer.data();
                recv_buffer_ptr_2 =
                    const_cast<scalar *>(mtx->get_const_values());
            } else {
                recv_buffer_ptr = const_cast<scalar *>(mtx->get_const_values());
            }

            if (comm_rank != rank) {
                comm.recv(device_exec, recv_buffer_ptr, comm_size, comm_rank,
                          tag);
                if (requires_host_buffer) {
                    auto host_buffer_view = gko::array<scalar>::view(
                        exec, comm_size, recv_buffer_ptr);
                    auto target_buffer_view = gko::array<scalar>::view(
                        device_exec, comm_size, recv_buffer_ptr_2);
                    target_buffer_view = host_buffer_view;
                }

            } else {
                // if data is already on this rank
                auto data_view = gko::array<scalar>::const_view(
                    exec, comm_size,
                    host_A->get_interface_data(host_interface_ctr));

                // TODO FIXME this needs target executor
                recv_buffer_ptr = const_cast<scalar *>(mtx->get_const_values());
                auto target_view = gko::array<scalar>::view(
                    device_exec, comm_size, recv_buffer_ptr);

                target_view = data_view;

                host_interface_ctr++;
                remain_host_interfaces--;
            }

            ctr++;
            // interface values need to be multiplied by -1
            using vec = gko::matrix::Dense<scalar>;
            recv_buffer_ptr = const_cast<scalar *>(mtx->get_const_values());
            auto neg_one = gko::initialize<vec>({-1.0}, device_exec);
            auto interface_dense = vec::create(
                device_exec,
                gko::dim<2>{static_cast<gko::size_type>(comm_size), 1},
                gko::array<scalar>::view(device_exec, comm_size,
                                         recv_buffer_ptr),
                1);

            interface_dense->scale(neg_one);
        }
    } else {
        // the non owner has send all its interfaces to owner
        // thus all values need to be communicated to the owner as well
        label num_interfaces = src_comm_pattern->target_ids.get_size();
        label tag = 0;
        for (int i = 0; i < num_interfaces; i++) {
            label comm_size =
                src_comm_pattern->target_sizes.get_const_data()[i];
            const scalar *send_buffer_ptr = host_A->get_interface_data(i);
            comm.send(device_exec, send_buffer_ptr, comm_size, owner_rank, tag);
        }
    }

    // reorder updated values
    if (owner) {
        // NOTE local sparsity size includes the interfaces
        using Coo = gko::matrix::Coo<scalar, label>;
        using dim_type = gko::dim<2>::dimension_type;

        std::shared_ptr<const Coo> local = gko::as<Coo>(
            gko::as<CombinationMatrix<Coo>>(dist_A->get_local_matrix())
                ->get_combination()
                ->get_operators()[0]);

        auto local_elements = local->get_num_stored_elements();
        local_ptr = const_cast<scalar *>(local->get_const_values());

        // TODO make sure this doesn't copy
        // create a non owning dense matrix of local_values
        auto row_collection = gko::share(gko::matrix::Dense<scalar>::create(
            device_exec, gko::dim<2>{static_cast<dim_type>(local_elements), 1},
            gko::array<scalar>::view(device_exec, local_elements, local_ptr),
            1));
        auto mapping_view = gko::array<label>::view(
            exec, local_elements, local_sparsity->ldu_mapping.get_data());


        // TODO this needs to copy ldu_mapping to the device
        auto dense_vec = row_collection->clone();
        // auto dense_vec =
        // gko::share(gko::matrix::Dense<scalar>::create(exec,
        // row_collection->get_size()));

        dense_vec->row_gather(&mapping_view, row_collection.get());
    }
};


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
static std::shared_ptr<
    RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>
RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    const ExecutorHandler &exec_handler, word matrix_format,
    const Repartitioner &repartitioner,
    std::shared_ptr<const HostMatrixWrapper> host_A)
{
    label rank = exec_handler.get_rank();
    auto exec = exec_handler.get_ref_exec();
    auto comm = *exec_handler.get_communicator().get();

    // repartition things here first by creating device_matrix_data on
    // device with correct size on each rank
    //
    // NOTE Start by copying first and repartition later
    auto local_sparsity = host_A->compute_local_sparsity(exec);
    auto non_local_sparsity = host_A->compute_non_local_sparsity(exec);

    auto src_comm_pattern = host_A->create_communication_pattern();
    auto repart_comm_pattern =
        repartitioner.repartition_comm_pattern(exec_handler, src_comm_pattern);

    // bool owner = repartitioner.is_owner(exec_handler);

    auto [repart_loc_sparsity, repart_non_loc_sparsity, local_interfaces] =
        repartitioner.repartition_sparsity(exec_handler, local_sparsity,
                                           non_local_sparsity);

    compress_cols(repart_non_loc_sparsity->col_idxs);

    // create vector of inner type linops
    // if fuse the vector contains only a single element
    // thus we can unwrap it.
    auto device_exec = exec_handler.get_device_exec();
    auto local_linops = detail::generate_inner_linops(
        matrix_format, false, device_exec, repart_loc_sparsity);
    auto non_local_linops = detail::generate_inner_linops(
        matrix_format, false, device_exec, repart_non_loc_sparsity);

    auto [send_counts, send_offsets, recv_sizes, recv_offsets] =
        repart_comm_pattern->send_recv_pattern();

    // recv_gather_idxs are send upon creation to ginkgo distributed matrix
    // to partner ranks. This sets the send_sizes_ on the partner ranks.
    // Thus recv_gather_idxs are local indices of comm partner rank of
    // interfaces.
    auto recv_gather_idxs =
        repart_comm_pattern->compute_recv_gather_idxs(exec_handler);

    auto global_rows = repartitioner.get_orig_partition()->get_size();
    gko::dim<2> global_dim{global_rows, global_rows};

    std::shared_ptr<dist_mtx> dist_A;
    bool fuse = false;
    if (fuse) {
    } else {
        dist_A = gko::share(dist_mtx::create(
            exec, comm, global_dim,
            gko::share(
                CombinationMatrix<gko::matrix::Coo<scalar, label>>::create(
                    exec, repart_loc_sparsity->dim, local_linops)),
            gko::share(
                CombinationMatrix<gko::matrix::Coo<scalar, label>>::create(
                    exec, repart_non_loc_sparsity->dim, non_local_linops)),
            recv_sizes, recv_offsets, recv_gather_idxs));
    }

    detail::update_impl(exec_handler, matrix_format, repartitioner, host_A,
                        dist_A, repart_loc_sparsity, repart_non_loc_sparsity,
                        src_comm_pattern, local_interfaces);

    return std::make_shared<RepartDistMatrix>(
        exec, comm, repartitioner.get_repart_dim(), dist_A->get_size(),
        std::move(dist_A), repart_loc_sparsity, repart_non_loc_sparsity,
        src_comm_pattern, local_interfaces);
}

template class RepartDistMatrix<scalar, label, label>;
