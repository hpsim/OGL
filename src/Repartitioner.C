// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/Repartitioner.H"

namespace detail {

std::tuple<std::vector<gko::span>, std::vector<label>, std::vector<label>> exchange_span_ranks(
    const ExecutorHandler &exec_handler, label ranks_per_gpu,
    const std::vector<gko::span> &spans, const std::vector<label> &ranks)
{
    auto comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, spans.size());

    std::vector<label> size{};

    for (auto &elem : spans) {
        size.push_back(elem.length());
    }

    auto gathered_size = gather_labels_to_owner(exec_handler, comm_pattern,
                                                size.data(), size.size());

    std::vector<gko::span> out_spans{};
    int count = 0;
    for (auto length : gathered_size) {
        out_spans.emplace_back(count, count + length);
        count += length;
    }

    auto new_ranks = gather_labels_to_owner(exec_handler, comm_pattern,
                                            ranks.data(), ranks.size());

    std::vector<label> origins{};
    if (gathered_size.size() > 0) {
    auto rank = exec_handler.get_rank();
    for (int i = 0; i<ranks_per_gpu; i++) {
        auto j = comm_pattern.recv_counts[i+rank];
        for (int k=0;k<j;k++) {
            origins.push_back(i+rank);
        }
    }
    }

    return {out_spans, new_ranks, origins};
}

template <typename T>
std::vector<T> apply_permutation(const std::vector<T> vec,
                                 const std::vector<label> &p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
                   [&](label i) { return vec[i]; });
    return sorted_vec;
}

template <typename T, typename Compare>
std::vector<label> sort_permutation(const std::vector<T> &vec, Compare compare)
{
    std::vector<label> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::stable_sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j) {
        return compare(vec[i], vec[j]);
    });
    return p;
}

}  // namespace detail

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
    std::shared_ptr<const SparsityPattern> src_local_pattern,
    std::shared_ptr<const SparsityPattern> src_non_local_pattern) const
{
    LOG_1(verbose_, "start repartition sparsity pattern")

    // helper function to simplify calling gather_labels_to_owner
    auto gather_closure = [exec_handler](auto &comm_pattern, auto &data,
                                         label offset) {
        return gather_labels_to_owner(exec_handler, comm_pattern,
                                      data.get_data(), data.get_size(), offset);
    };

    auto exec = exec_handler.get_ref_exec();
    auto comm = *exec_handler.get_communicator().get();
    label rank = exec_handler.get_rank();
    label owner_rank = get_owner_rank(exec_handler);
    label ranks_per_gpu = ranks_per_gpu_;

    // early return if no repartitioning requested
    if (ranks_per_gpu == 1) {
        std::vector<std::pair<bool, label>> ret;
        for (auto comm_rank : src_non_local_pattern->rank) {
            ret.emplace_back(false, rank);
        }
        return std::make_tuple<std::shared_ptr<SparsityPattern>,
                               std::shared_ptr<SparsityPattern>,
                               std::vector<std::pair<bool, label>>>(
            std::make_shared<SparsityPattern>(*src_local_pattern.get()),
            std::make_shared<SparsityPattern>(*src_non_local_pattern.get()),
            std::move(ret));
    }

    // Step 1. gather all local sparsity pattern to owner rank
    // this is a all in collected into std::vectors, thus we can
    // append all the interfaces which became local after repartitioning
    auto local_comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, src_local_pattern->num_nnz);

    // row/column index offset relative to its owner rank
    auto offset = orig_partition_->get_range_bounds()[rank] -
                  orig_partition_->get_range_bounds()[owner_rank];

    auto tmp_local_rows =
        gather_closure(local_comm_pattern, src_local_pattern->row_idxs, offset);
    auto tmp_local_cols =
        gather_closure(local_comm_pattern, src_local_pattern->col_idxs, offset);
    auto tmp_local_mapping =
        gather_closure(local_comm_pattern, src_local_pattern->ldu_mapping, 0);

    if (is_owner(exec_handler)) {
        make_ldu_mapping_consecutive(local_comm_pattern, tmp_local_mapping,
                                     rank, ranks_per_gpu);
    }
    std::vector<gko::span> tmp_local_span{gko::span{0, tmp_local_rows.size()}};
    std::vector<label> tmp_local_rank{rank};

    gko::dim<2> tmp_local_dim = (is_owner(exec_handler))
                                    ? compute_dimensions(tmp_local_rows)
                                    : gko::dim<2>{0, 0};

    // Done with Step 1. Next we gather all non local sparsity. Again
    // all of it is collected into temporary vectors first.

    auto non_local_comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, src_non_local_pattern->num_nnz);

    auto tmp_non_local_rows = gather_closure(
        non_local_comm_pattern, src_non_local_pattern->row_idxs, offset);

    auto [new_spans, new_ranks, tmp_non_local_origin] = detail::exchange_span_ranks(
        exec_handler, ranks_per_gpu_, src_non_local_pattern->spans,
        src_non_local_pattern->rank);

    auto tmp_non_local_cols = detail::convert_to_global(
        orig_partition_,
        gather_closure(non_local_comm_pattern, src_non_local_pattern->col_idxs,
                       0),
        new_spans, new_ranks);

    std::vector<label> tmp_non_local_mapping =
        std::vector<label>(tmp_non_local_rows.size());

    auto is_local = build_non_local_interfaces(
        exec_handler, orig_partition_, tmp_local_rows, tmp_local_cols,
        tmp_local_mapping, tmp_local_rank, tmp_local_span, tmp_non_local_rows,
        tmp_non_local_cols, tmp_non_local_mapping, new_ranks,
        tmp_non_local_origin, new_spans);

    // TODO FIXME
    gko::dim<2> tmp_non_local_dim{0, 0};

    if (is_owner(exec_handler)) {
        auto local_pattern = std::make_shared<SparsityPattern>(
            exec, tmp_local_dim, tmp_local_rows, tmp_local_cols,
            tmp_local_mapping, tmp_local_span, tmp_local_rank);
        auto non_local_pattern = std::make_shared<SparsityPattern>(
            exec, tmp_non_local_dim, tmp_non_local_rows, tmp_non_local_cols,
            tmp_non_local_mapping, new_spans, new_ranks);

        return std::make_tuple<std::shared_ptr<SparsityPattern>,
                               std::shared_ptr<SparsityPattern>,
                               std::vector<std::pair<bool, label>>>(
            std::move(local_pattern), std::move(non_local_pattern), {});
    } else {
        return std::make_tuple<std::shared_ptr<SparsityPattern>,
                               std::shared_ptr<SparsityPattern>,
                               std::vector<std::pair<bool, label>>>(
            std::make_shared<SparsityPattern>(exec),
            std::make_shared<SparsityPattern>(exec), {});
    }
    // auto span_comm_pattern = compute_gather_to_owner_counts(
    //     exec_handler, ranks_per_gpu, src_non_local_pattern->spans.size());
    // // build vector with locality information
    // std::vector<std::pair<bool, label>> locality;
    // label ctr{0};

    // if (is_owner(exec_handler)) {
    //     for (int i = 0; i < ranks_per_gpu; i++) {
    //         label comm_rank = rank + i;
    //         label num_elems = span_comm_pattern.recv_counts[comm_rank];
    //         for (int j = 0; j < num_elems; j++) {
    //             locality.emplace_back(is_local[ctr], comm_rank);
    //             ctr++;
    //         }
    //     }
    // }
    // // All interfaces that are in is_local need to be accounted for
    // ASSERT_EQ(ctr, is_local.size());

    // // TODO FIXME the correct way would be to check the communication
    // // pattern where a particular face is in the send idxs
    // // since we have already the row id of the other side
    // // it should be doable. Or alternatively we know that we
    // // keep an interface together thus we can just count the idx up to the
    // // size. But we have to make sure that the interfaces are in the same
    // // order on both communication sides.
    // for (int i = 0; i < gathered_non_local.col_idxs.get_size(); i++) {
    //     gathered_non_local.col_idxs.get_data()[i] = i;
    // }

    // LOG_1(verbose_, "done repartition sparsity pattern")
    // if (is_owner(exec_handler)) {
    //     FatalErrorInFunction << "Not implemented" << abort(FatalError);
    //     // auto new_local_spars_pattern = std::make_shared<SparsityPattern>(
    //     //     exec, merged_local_dim, merged_local);
    //     //
    //     // auto new_non_local_spars_pattern =
    //     //     std::make_shared<SparsityPattern>(
    //     //         exec,
    //     //         gko::dim<2>{merged_local_dim[0],
    //     //                     gathered_non_local.row_idxs.get_size()},
    //     //         gathered_non_local);
    //     //
    //     //         return std::make_tuple<std::shared_ptr<SparsityPattern>,
    //     //                                std::shared_ptr<SparsityPattern>,
    //     //                                std::vector<std::pair<bool,
    //     //                                label>>>(
    //     //             std::move(new_local_spars_pattern),
    //     //             std::move(new_non_local_spars_pattern),
    //     //             std::move(locality));
    // } else {
    //     // auto new_local_spars_pattern =
    //     //     std::make_shared<SparsityPattern>(exec);
    //     //
    //     // auto new_non_local_spars_pattern =
    //     //     std::make_shared<SparsityPattern>(exec);
    //     //
    //     // // FIXME
    //     // // std::cout << __FILE__ << ":" << __LINE__ << " rank " <<
    //     // // comm.rank() <<
    //     // //     " merged_local dim " <<
    //     // new_local_spars_pattern->dim =
    //     //     gko::dim<2>{0, 0};  // merged_local_dim;
    //     // new_non_local_spars_pattern->dim =
    //     //     gko::dim<2>{0, 0};  // merged_local_dim;
    //     //
    //     //
    //     FatalErrorInFunction << "Not implemented" << abort(FatalError);
    //     // return std::make_tuple<std::shared_ptr<SparsityPattern>,
    //     //                        std::shared_ptr<SparsityPattern>,
    //     //                        std::vector<std::pair<bool, label>>>(
    //     //     std::move(new_local_spars_pattern),
    //     //     std::move(new_non_local_spars_pattern),
    //     //     std::vector<std::pair<bool, label>>{});
    // }
    // // TODO Some old documentation see what is still needed
    // // add interfaces
    // // to add interfaces we go through the non_local_sparsity
    // // pattern and check if interface is still non_local
    // //
    // // iterate interfaces and send to owner.
    // // on owner decide to move to local or non_local
    // // for sending to owner we can also use the all_to_all_v
    // // approach we than have
    // //
    // // row [ 4 , 8,  12 | 16 , 17, 18 ] <- local row
    // // col [ 1 , 2, 3 | 1, 2, 3 ] <- just the interface ctr
    // // from repartition_comm_pattern we could get
    // //
    // // or we split local / non-local first
    // // for this we could store ranges when computing
    // // repartition_comm_pattern
    // //
}

std::vector<bool> Repartitioner::build_non_local_interfaces(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
        partition,
    std::vector<label> &local_rows, std::vector<label> &local_cols,
    std::vector<label> &local_mapping, std::vector<label> &local_ranks,
    std::vector<gko::span> &local_spans, std::vector<label> &non_local_rows,
    std::vector<label> &non_local_cols, std::vector<label> &non_local_mapping,
    std::vector<label> &non_local_ranks,
    std::vector<label> &non_local_rank_origin,
    std::vector<gko::span> &non_local_spans) const
{
    auto rank = exec_handler.get_rank();
    std::vector<bool> is_local;
    std::vector<label> mark_keep;

    for (int i = 0; i < non_local_spans.size(); i++) {
        auto [begin, end] = non_local_spans[i];
        bool local = reparts_to_local(exec_handler, non_local_ranks[i]);
        is_local.push_back(local);

        if (local) {
            local_rows.insert(local_rows.end(), non_local_rows.data() + begin,
                              non_local_rows.data() + end);

            std::vector<label> tmp_rank_local_cols(
                non_local_cols.data() + begin,
                non_local_cols.data() + end);

            detail::convert_to_local(partition, tmp_rank_local_cols, rank);

            local_cols.insert(local_cols.end(), tmp_rank_local_cols.begin(),
                              tmp_rank_local_cols.end());
            local_mapping.insert(local_mapping.end(),
                                 non_local_mapping.data() + begin,
                                 non_local_mapping.data() + end);

            gko::size_type rows_start = local_rows.size();
            local_spans.emplace_back(
                rows_start,
                          static_cast<gko::size_type>(end - begin));

            local_ranks.push_back(non_local_rank_origin[i]);
        } else {
            mark_keep.push_back(i);
        }
    }

    // remove data from non_local vectors
    if (mark_keep.size() == 0) {
    non_local_rows.clear();
    non_local_cols.clear();
    non_local_mapping.clear();
    non_local_ranks.clear();
    non_local_spans.clear();
        } else {
    std::vector<label> copy_rows, copy_cols, copy_mapping, copy_ranks;
    std::vector<gko::span> copy_spans;
    for (label i : mark_keep) {
        auto [begin, end] = non_local_spans[i];
        copy_rows.insert(copy_rows.end(),
                         non_local_rows.data() + begin,
                         non_local_rows.data() + end);
        copy_cols.insert(copy_cols.end(),
                         non_local_cols.data() + begin,
                         non_local_cols.data() + end);
        copy_mapping.insert(copy_mapping.end(),
                            non_local_mapping.data() + begin,
                            non_local_mapping.data() + end);
        copy_ranks.push_back(non_local_ranks[i]);
        copy_spans.emplace_back(non_local_spans[i].begin,
                                non_local_spans[i].end);
    }

    non_local_rows = copy_rows;
    non_local_cols = copy_cols;
    non_local_mapping = copy_mapping;
    non_local_ranks = copy_ranks;
    non_local_spans.clear();
    for (auto &span : copy_spans) {
        non_local_spans.emplace_back(span.begin, span.end);
    }
    }

    return is_local;
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
    auto p = detail::sort_permutation(gathered_target_ids,
                                      [](label a, label b) { return a < b; });

    target_sizes = detail::apply_permutation(gathered_target_sizes, p);
    target_ids = detail::apply_permutation(gathered_target_ids, p);
    send_idxs = detail::apply_permutation(send_idxs, p);

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
