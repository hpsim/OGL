// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/Repartitioner.hpp"

label Repartitioner::compute_repart_size(label local_size, label ranks_per_gpu,
                                         const ExecutorHandler &exec_handler)
{
    if (ranks_per_gpu == 1) {
        return local_size;
    }

    auto all_to_all_pattern =
        compute_gather_to_owner_counts(exec_handler, ranks_per_gpu, local_size);

    return all_to_all_pattern.recv_offsets.back();
}

std::pair<std::shared_ptr<SparsityPattern>, std::shared_ptr<SparsityPattern>>
Repartitioner::repartition_sparsity(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<SparsityPattern> src_local_pattern,
    std::shared_ptr<SparsityPattern> src_non_local_pattern) const
{
    LOG_1(verbose_, "start repartition sparsity pattern")


    auto exec = exec_handler.get_ref_exec();
    auto comm = *exec_handler.get_communicator().get();
    label rank = exec_handler.get_rank();
    label owner_rank = get_owner_rank(exec_handler);
    label ranks_per_gpu = ranks_per_gpu_;

    // early return if no repartitioning requested
    if (ranks_per_gpu == 1) {
        LOG_1(verbose_, "done repartition sparsity pattern")
        return {src_local_pattern, src_non_local_pattern};
    }

    // helper function to simplify calling gather_labels_to_owner
    auto gather_closure = [exec_handler](auto &comm_pattern, auto &data,
                                         label offset) {
        return gather_labels_to_owner(exec_handler, comm_pattern, data.data(),
                                      data.size(), offset);
    };

    // row/column index offset relative to its owner rank
    auto offset = orig_partition_->get_range_bounds()[rank] -
                  orig_partition_->get_range_bounds()[owner_rank];

    auto gather_vector = [exec_handler, ranks_per_gpu, gather_closure](
                             size_t length, label offset, auto &in) {
        auto comm_pattern =
            compute_gather_to_owner_counts(exec_handler, ranks_per_gpu, length);

        std::vector<label> tmp;
        tmp.reserve(length);
        for (auto &i : in) {
            for (auto j : i) {
                tmp.push_back(j);
            }
        }

        return gather_closure(comm_pattern, tmp, offset);
    };


    auto create_sparsity = [&](const auto &in_sparsity, auto &rows, auto &cols,
                               auto &map, bool fuse) {
        auto lengths_tmp = in_sparsity->get_lengths();
        auto size_comm_pattern = compute_gather_to_owner_counts(
            exec_handler, ranks_per_gpu, lengths_tmp.size());

        auto lengths = gather_closure(size_comm_pattern, lengths_tmp, 0);

        auto orig_ids =
            gather_closure(size_comm_pattern, in_sparsity->get_id(), 0);
        auto orig_ranks =
            gather_closure(size_comm_pattern, in_sparsity->get_orig_rank(), 0);
        auto comm_ranks =
            gather_closure(size_comm_pattern, in_sparsity->get_comm_rank(), 0);

        auto sparsity = std::make_shared<SparsityPattern>();
        size_t ctr{0};
        for (size_t i = 0; i < lengths.size(); i++) {
            auto length = lengths[i];
            sparsity->insert_interface(
                std::vector<label>(rows.data() + ctr,
                                   rows.data() + ctr + length),
                std::vector<label>(cols.data() + ctr,
                                   cols.data() + ctr + length),
                std::vector<label>(map.data() + ctr, map.data() + ctr + length),
                orig_ids[i], orig_ranks[i], comm_ranks[i]);
            ctr += length;
        }
        return sparsity;
    };

    // get and exchange sizes of individidual local vectors
    size_t loc_nnz = src_local_pattern->get_nnz();
    auto loc_map = gather_vector(loc_nnz, 0, src_local_pattern->get_map());
    auto loc_col =
        gather_vector(loc_nnz, offset, src_local_pattern->get_cols());
    auto loc_row =
        gather_vector(loc_nnz, offset, src_local_pattern->get_rows());
    auto ret_local_sparsity =
        create_sparsity(src_local_pattern, loc_row, loc_col, loc_map, true);

    size_t non_loc_nnz = src_non_local_pattern->get_nnz();
    auto non_loc_map =
        gather_vector(non_loc_nnz, 0, src_non_local_pattern->get_map());
    auto non_loc_row =
        gather_vector(non_loc_nnz, offset, src_non_local_pattern->get_rows());
    // non_loc_cols are global idx so no offset needed
    auto non_loc_col =
        gather_vector(non_loc_nnz, 0, src_non_local_pattern->get_cols());
    auto ret_non_local_sparsity = create_sparsity(
        src_non_local_pattern, non_loc_row, non_loc_col, non_loc_map, false);

    for (auto &comm_rank : ret_non_local_sparsity->get_comm_rank()) {
        comm_rank = compute_owner_rank(comm_rank, ranks_per_gpu);
    }

    auto global_to_local_offset = orig_partition_->get_range_bounds()[rank];
    auto convert_to_local = [global_to_local_offset](std::vector<label> &&in) {
        std::vector<label> out(in);
        for (auto &val : out) {
            val = val - global_to_local_offset;
        }
        return out;
    };

    // non owning ranks seems to have wrong non-local size
    if (ret_local_sparsity->get_nnz() != 0) {
        ret_local_sparsity->move_interface(ret_non_local_sparsity, rank,
                                           convert_to_local);
    }
    return {ret_local_sparsity, ret_non_local_sparsity};
}

std::shared_ptr<const CommunicationPattern>
Repartitioner::repartition_comm_pattern(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const CommunicationPattern> src_comm_pattern) const
{
    if (ranks_per_gpu_ == 1) {
        return src_comm_pattern;
    }

    // using comm_size_type = label;
    auto exec = exec_handler.get_ref_exec();
    auto comm = src_comm_pattern->get_comm();

    label rank = comm.rank();
    bool owner = is_owner(exec_handler);

    // Step 1. Check if communication partner is non-local after
    // repartitioning. If it is non-local we keep it. Otherwise communication
    // partners that are local after repartitioning can be discarded. Here
    // non-local means: the communication target rank id is != repartitioned
    // rank id
    std::vector<label> target_ids{};
    std::vector<label> target_sizes{};
    std::vector<std::vector<label>> send_idxs;
    label communication_partner = src_comm_pattern->target_ids.size();
    for (int i = 0; i < communication_partner; i++) {
        label target_id = src_comm_pattern->target_ids.data()[i];
        if (!reparts_to_local(exec_handler, target_id)) {
            // communication pattern is non local, hence we keep it
            // after repartitioning we now have to communicate with a
            // different rank store new owner rank to which repart_target_*
            // needs to be send to
            target_ids.push_back(get_owner_rank(target_id));
            target_sizes.push_back(src_comm_pattern->target_sizes.data()[i]);
            send_idxs.push_back(src_comm_pattern->send_idxs[i]);
        }
    }

    // send all remaining non local ids and sizes to the new
    // owner rank
    auto comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu_, target_ids.size());

    auto gathered_target_ids = gather_labels_to_owner(
        exec_handler, comm_pattern, target_ids.data(), target_ids.size());

    auto gathered_target_sizes = gather_labels_to_owner(
        exec_handler, comm_pattern, target_sizes.data(), target_sizes.size());

    // next the send_ixs need to be updated we send them piecewise since
    // the send_idxs are a vector of gko::arrays
    if (owner) {
        label recv_ctr = comm_pattern.recv_counts[rank];
        // retrieved from i-th neighbor
        for (int i = 1; i < ranks_per_gpu_; i++) {
            // how many gko::arrays to with send_indexes to receive
            // from i-th neighbor
            label recv_count = comm_pattern.recv_counts[rank + i];

            for (int j = 0; j < recv_count; j++) {
                auto target_size = gathered_target_sizes[j + recv_ctr];
                std::vector<label> recv_buffer(target_size);

                comm.recv(exec, recv_buffer.data(), target_size, rank + i,
                          rank);

                // the new offset is
                auto offset =
                    get_orig_partition()->get_range_bounds()[rank + i] -
                    get_orig_partition()->get_range_bounds()[rank];

                std::transform(recv_buffer.begin(), recv_buffer.end(),
                               recv_buffer.begin(),
                               [&](label idx) { return idx + offset; });

                // auto target_id = gathered_target_ids[j + owner_recv_counts];
                send_idxs.emplace_back(recv_buffer);
            }
            recv_ctr += recv_count;
        }
    } else {
        label owner = get_owner_rank(exec_handler);
        for (int i = 0; i < comm_pattern.send_counts[owner]; i++) {
            comm.send(exec, send_idxs[i].data(), send_idxs[i].size(), owner,
                      owner);
        }
    }

    // clear communication neighbors on non owning rank
    if (!owner) {
        // TODO NOTE should it be gathered_target_ids and is it needed?
        target_ids.clear();
        target_sizes.clear();
        send_idxs.clear();
    }

    // early return if no communication partners are left
    if (gathered_target_ids.size() == 0) {
        return std::make_shared<CommunicationPattern>(exec_handler, target_ids,
                                                      send_idxs);
    }

    // merge communication
    // we now might have communication partners multiple times, thus we can
    // merge them sort repart_target_ids and use the sorting for
    // target_sizes and repart_send_idxs
    std::vector<label> merged_target_ids{};
    std::vector<label> merged_target_sizes{};
    std::vector<std::vector<label>> merged_send_idxs{};
    auto p = detail::sort_permutation(gathered_target_ids,
                                      [](label a, label b) { return a < b; });

    target_sizes = detail::apply_permutation(gathered_target_sizes, p);
    target_ids = detail::apply_permutation(gathered_target_ids, p);
    send_idxs = detail::apply_permutation(send_idxs, p);

    // Step 4.
    // Merge communication pattern with corresponding neighbours
    merged_target_ids.push_back(target_ids[0]);
    merged_target_sizes.push_back(target_sizes[0]);
    merged_send_idxs.emplace_back(std::vector<label>(send_idxs[0]));

    for (size_t i = 1; i < target_ids.size(); i++) {
        // communicates with same target rank
        // thus we have only have to adapt the number
        // of elements and the send_ixs
        auto *send_idx_begin = send_idxs[i].data();
        auto *send_idx_end = send_idxs[i].data() + send_idxs[i].size();
        if (target_ids[i] == merged_target_ids.back()) {
            merged_target_sizes.back() += target_sizes[i];
            merged_send_idxs.back().insert(merged_send_idxs.back().end(),
                                           send_idx_begin, send_idx_end);
        } else {
            merged_target_ids.push_back(target_ids[i]);
            merged_target_sizes.push_back(target_sizes[i]);
            merged_send_idxs.emplace_back(
                std::vector<label>(send_idx_begin, send_idx_end));
        }
    }

    // recompute send_idxs
    send_idxs.clear();

    for (size_t i = 0; i < merged_target_ids.size(); i++) {
        // label target_id = merged_target_ids[i];
        send_idxs.emplace_back(merged_send_idxs[i]);
    }

    return std::make_shared<CommunicationPattern>(exec_handler,
                                                  merged_target_ids, send_idxs);
}
