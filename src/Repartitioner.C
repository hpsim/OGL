// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/Repartitioner.H"


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

std::tuple<std::shared_ptr<SparsityPattern>, std::shared_ptr<SparsityPattern>,
           std::vector<std::pair<bool, label>>>
Repartitioner::repartition_sparsity(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<SparsityPattern> src_local_pattern,
    std::shared_ptr<SparsityPattern> src_non_local_pattern) const
{
    FatalErrorInFunction << "Not implemented" << exit(FatalError);
}

std::pair<SparsityPattern, std::vector<bool>>
Repartitioner::build_non_local_interfaces(const ExecutorHandler &exec_handler,
                                          SparsityPattern &loc,
                                          const SparsityPattern &non_loc) const
{
    FatalErrorInFunction << "Not implemented" << exit(FatalError);
}


std::shared_ptr<const CommunicationPattern>
Repartitioner::repartition_comm_pattern(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const CommunicationPattern> src_comm_pattern,
    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
        partition) const
{
    if (ranks_per_gpu_ == 1) {
        return src_comm_pattern;
    }

    using comm_size_type = label;
    auto exec = src_comm_pattern->target_ids.get_executor();
    auto comm = src_comm_pattern->get_comm();

    label rank = comm.rank();
    bool owner = is_owner(exec_handler);

    // Step 1. Check if communication partner is non-local after
    // repartitioning. If it is non-local we keep it. Otherwise communcation
    // partners that are local after repartitioning can be discarded. Here
    // non-local means: the communication target rank id is != repartitioned
    // rank id
    std::vector<label> target_ids{};
    std::vector<label> target_sizes{};
    std::vector<std::pair<gko::array<label>, comm_size_type>> send_idxs;
    label communication_partner = src_comm_pattern->target_ids.get_size();
    for (int i = 0; i < communication_partner; i++) {
        label target_id = src_comm_pattern->target_ids.get_const_data()[i];
        if (!reparts_to_local(exec_handler, target_id)) {
            // communication pattern is non local, hence we keep it
            // after repartitioning we now have to communicate with a
            // different rank store new owner rank to which repart_target_*
            // needs to be send to
            target_ids.push_back(get_owner_rank(target_id));
            target_sizes.push_back(
                src_comm_pattern->target_sizes.get_const_data()[i]);
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
        label owner_recv_counts = comm_pattern.recv_counts[rank];
        // retrieved from i-th neighbor
        for (int i = 1; i < ranks_per_gpu_; i++) {
            // how many gko::arrays to with send_indexes to receive
            // from i-th neighbor
            label recv_count = comm_pattern.recv_counts[rank + i];
            for (int j = 0; j < recv_count; j++) {
                auto target_size = gathered_target_sizes[j + owner_recv_counts];
                std::vector<label> recv_buffer(target_size);

                comm.recv(exec, recv_buffer.data(), target_size, rank + i,
                          rank);

                // the new offset is
                auto offset = partition->get_range_bounds()[rank + i] -
                              partition->get_range_bounds()[rank];

                std::transform(recv_buffer.begin(), recv_buffer.end(),
                               recv_buffer.begin(),
                               [&](label idx) { return idx + offset; });

                auto target_id = gathered_target_ids[j + owner_recv_counts];
                send_idxs.push_back({gko::array<label>{
                                         exec,
                                         recv_buffer.begin(),
                                         recv_buffer.end(),
                                     },
                                     target_id});
            }
        }
    } else {
        label owner = get_owner_rank(exec_handler);
        for (int i = 0; i < comm_pattern.send_counts[owner]; i++) {
            auto send_buffer = send_idxs[i].first;
            comm.send(exec, send_buffer.get_const_data(),
                      send_buffer.get_size(), owner, owner);
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
        return std::make_shared<CommunicationPattern>(
            exec_handler,
            gko::array<comm_size_type>{exec, target_ids.begin(),
                                       target_ids.end()},
            gko::array<comm_size_type>{exec, target_sizes.begin(),
                                       target_sizes.end()},

            send_idxs);
    }

    // merge communication
    // we now might have communication partners multiple times, thus we can
    // merge them sort repart_target_ids and use the sorting for
    // target_sizes and repart_send_idxs
    std::vector<label> merged_target_ids{};
    std::vector<label> merged_target_sizes{};
    std::vector<std::vector<label>> merged_send_idxs{};
    auto p = sort_permutation(gathered_target_ids,
                              [](label a, label b) { return a < b; });

    target_sizes = apply_permutation(gathered_target_sizes, p);
    target_ids = apply_permutation(gathered_target_ids, p);
    send_idxs = apply_permutation(send_idxs, p);

    // Step 4.
    // Merge communication pattern with corresponding neighbours
    merged_target_ids.push_back(target_ids[0]);
    merged_target_sizes.push_back(target_sizes[0]);
    merged_send_idxs.emplace_back(std::vector<label>(
        send_idxs[0].first.get_data(),
        send_idxs[0].first.get_data() + send_idxs[0].first.get_size()));


    for (int i = 1; i < target_ids.size(); i++) {
        // communicates with same target rank
        // thus we have only have to adapt the number
        // of elements and the send_ixs
        auto *send_idx_begin = send_idxs[i].first.get_data();
        auto *send_idx_end =
            send_idxs[i].first.get_data() + send_idxs[i].first.get_size();
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

    for (int i = 0; i < merged_target_ids.size(); i++) {
        label target_id = merged_target_ids[i];
        send_idxs.emplace_back(std::pair<gko::array<label>, comm_size_type>{
            gko::array<label>{exec, merged_send_idxs[i].begin(),
                              merged_send_idxs[i].end()},
            target_id});
    }

    return std::make_shared<CommunicationPattern>(
        exec_handler,
        gko::array<comm_size_type>{exec, merged_target_ids.begin(),
                                   merged_target_ids.end()},
        gko::array<comm_size_type>{exec, merged_target_sizes.begin(),
                                   merged_target_sizes.end()},
        send_idxs);
}
