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
    LOG_1(verbose_, "start repartition sparsity pattern")
    // 1. obtain send recv sizes vector
    // here we can reuse code from repartition_comm_pattern
    //
    // 2. initialize send and recv buffer
    // if we keep interfaces separated row and col indices can be
    //
    // 3. ldu mapping needs to be update?
    // 4. dim needs to be updated
    // this is not implemented yet, but we can't fail here for
    // debugging reasons
    auto exec = exec_handler.get_ref_exec();
    auto comm = *exec_handler.get_communicator().get();
    label rank = get_rank(exec_handler);
    label owner_rank = get_owner_rank(exec_handler);
    label ranks_per_gpu = ranks_per_gpu_;
    // TODO dont copy
    // if (ranks_per_gpu == 1) {
    //     std::vector<std::pair<bool, label>> ret;
    //     for (auto comm_rank : src_non_local_pattern->rank) {
    //         ret.emplace_back(false, rank);
    //     }
    //     return std::make_tuple<std::shared_ptr<SparsityPattern>,
    //                            std::shared_ptr<SparsityPattern>,
    //                            std::vector<std::pair<bool, label>>>(
    //         std::make_shared<SparsityPattern>(src_local_pattern),
    //         std::make_shared<SparsityPattern>(src_non_local_pattern),
    //         std::move(ret));
    // }

    auto local_comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, src_local_pattern->num_nnz);


    auto offset = orig_partition_->get_range_bounds()[rank] -
                  orig_partition_->get_range_bounds()[owner_rank];

    auto gather_closure = [&](auto &comm_pattern, auto &data, label offset) {
        return gather_labels_to_owner(exec_handler, comm_pattern,
                                      data.get_data(), data.get_size(), offset);
    };

    SparsityPattern merged_local{
        gather_closure(local_comm_pattern, src_local_pattern->row_idxs, offset),
        gather_closure(local_comm_pattern, src_local_pattern->col_idxs, offset),
        gather_closure(local_comm_pattern, src_local_pattern->ldu_mapping, 0),
        std::vector{rank}};

    if (is_owner(exec_handler)) {
        make_ldu_mapping_consecutive(
            local_comm_pattern, merged_local.ldu_mapping, rank, ranks_per_gpu);
    }

    auto get_back = [](const gko::array<label> &in) {
        return *(in.get_const_data() + in.get_size());
    };

    label rows =
        (is_owner(exec_handler)) ? get_back(merged_local.row_idxs) + 1 : 0;
    gko::dim<2> merged_local_dim{static_cast<gko::size_type>(rows),
                                 static_cast<gko::size_type>(rows)};

    auto non_local_comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, src_non_local_pattern->num_nnz);

    std::vector<label> spans_begin;
    std::vector<label> spans_end;
    auto span_comm_pattern = compute_gather_to_owner_counts(
        exec_handler, ranks_per_gpu, src_non_local_pattern->spans.size());
    for (auto elem : src_non_local_pattern->spans) {
        spans_begin.push_back(elem.begin);
        spans_end.push_back(elem.end);
    }

    // the non local cols are in local idx of other side
    // thus we need the new offset of the other side
    // NOTE TODO this modifies src_non_local_pattern better do the offsets
    // on the owner side after gathering
    for (label i = 0; i < src_non_local_pattern->rank.size(); i++) {
        auto comm_rank = src_non_local_pattern->rank[i];
        auto [begin, end] = src_non_local_pattern->spans[i];
        label local_offset = orig_partition_->get_range_bounds()[comm_rank] -
                             orig_partition_->get_range_bounds()[owner_rank];
        auto *data = src_non_local_pattern->col_idxs.get_data() + begin;
        auto size = end - begin;
        std::transform(data, data + size, data,
                       [&](label idx) { return idx + local_offset; });
    }

    SparsityPattern merged_non_local{
        gather_closure(non_local_comm_pattern, src_non_local_pattern->row_idxs,
                       offset),
        gather_closure(non_local_comm_pattern, src_non_local_pattern->col_idxs,
                       0),
        gather_closure(non_local_comm_pattern, src_non_local_pattern->row_idxs,
                       0),
        // {},
        // this doesn't exist
        // gather_labels_to_owner(exec_handler, span_comm_pattern,
        // spans_begin.size(),
        //                spans_begin.data()),
        //  gather_labels_to_owner(exec_handler, span_comm_pattern,
        //  spans_end.size(),
        //                spans_end.data()),
        gather_labels_to_owner(exec_handler, span_comm_pattern,
                               src_non_local_pattern->rank.data(),
                               src_non_local_pattern->rank.size())};

    if (is_owner(exec_handler)) {
        FatalErrorInFunction << "Not implemented" << abort(FatalError);
        //  make_begin_and_ends_consecutive(
        //      span_comm_pattern, merged_non_local.spans,
        //     rank, ranks_per_gpu);
    }

    auto [gathered_non_local, is_local] = build_non_local_interfaces(
        exec_handler, merged_local, merged_non_local);

    // build vector with locality information
    std::vector<std::pair<bool, label>> locality;
    label ctr{0};

    if (is_owner(exec_handler)) {
        for (int i = 0; i < ranks_per_gpu; i++) {
            label comm_rank = rank + i;
            label num_elems = span_comm_pattern.recv_counts[comm_rank];
            for (int j = 0; j < num_elems; j++) {
                locality.emplace_back(is_local[ctr], comm_rank);
                ctr++;
            }
        }
    }
    // All interfaces that are in is_local need to be accounted for
    ASSERT_EQ(ctr, is_local.size());

    // TODO FIXME the correct way would be to check the communication
    // pattern where a particular face is in the send idxs
    // since we have already the row id of the other side
    // it should be doable. Or alternatively we know that we
    // keep an interface together thus we can just count the idx up to the
    // size. But we have to make sure that the interfaces are in the same
    // order on both communication sides.
    for (int i = 0; i < gathered_non_local.col_idxs.get_size(); i++) {
        gathered_non_local.col_idxs.get_data()[i] = i;
    }

    LOG_1(verbose_, "done repartition sparsity pattern")
    if (is_owner(exec_handler)) {
        FatalErrorInFunction << "Not implemented" << abort(FatalError);
        // auto new_local_spars_pattern = std::make_shared<SparsityPattern>(
        //     exec, merged_local_dim, merged_local);
        //
        // auto new_non_local_spars_pattern =
        //     std::make_shared<SparsityPattern>(
        //         exec,
        //         gko::dim<2>{merged_local_dim[0],
        //                     gathered_non_local.row_idxs.get_size()},
        //         gathered_non_local);
        //
        //         return std::make_tuple<std::shared_ptr<SparsityPattern>,
        //                                std::shared_ptr<SparsityPattern>,
        //                                std::vector<std::pair<bool,
        //                                label>>>(
        //             std::move(new_local_spars_pattern),
        //             std::move(new_non_local_spars_pattern),
        //             std::move(locality));
    } else {
        // auto new_local_spars_pattern =
        //     std::make_shared<SparsityPattern>(exec);
        //
        // auto new_non_local_spars_pattern =
        //     std::make_shared<SparsityPattern>(exec);
        //
        // // FIXME
        // // std::cout << __FILE__ << ":" << __LINE__ << " rank " <<
        // // comm.rank() <<
        // //     " merged_local dim " <<
        // new_local_spars_pattern->dim =
        //     gko::dim<2>{0, 0};  // merged_local_dim;
        // new_non_local_spars_pattern->dim =
        //     gko::dim<2>{0, 0};  // merged_local_dim;
        //
        //
        FatalErrorInFunction << "Not implemented" << abort(FatalError);
        // return std::make_tuple<std::shared_ptr<SparsityPattern>,
        //                        std::shared_ptr<SparsityPattern>,
        //                        std::vector<std::pair<bool, label>>>(
        //     std::move(new_local_spars_pattern),
        //     std::move(new_non_local_spars_pattern),
        //     std::vector<std::pair<bool, label>>{});
    }
    // TODO Some old documentation see what is still needed
    // add interfaces
    // to add interfaces we go through the non_local_sparsity
    // pattern and check if interface is still non_local
    //
    // iterate interfaces and send to owner.
    // on owner decide to move to local or non_local
    // for sending to owner we can also use the all_to_all_v
    // approach we than have
    //
    // row [ 4 , 8,  12 | 16 , 17, 18 ] <- local row
    // col [ 1 , 2, 3 | 1, 2, 3 ] <- just the interface ctr
    // from repartition_comm_pattern we could get
    //
    // or we split local / non-local first
    // for this we could store ranges when computing
    // repartition_comm_pattern
    //
}

std::pair<SparsityPattern, std::vector<bool>>
Repartitioner::build_non_local_interfaces(const ExecutorHandler &exec_handler,
                                          SparsityPattern &loc,
                                          const SparsityPattern &non_loc) const
{
    LOG_1(verbose_, "start build non local interfaces")
    std::vector<label> rows, cols, ldu_mapping, ranks, begins, ends;
    std::vector<bool> is_local;
    label merged_ranks_size = non_loc.rank.size();

    // TODO dont modify SparsityPattern but create vector
    // and use the vector variant
    //   for (label i = 0; i < merged_ranks_size; i++) {
    //       // these are the begin ends before merging they need to be
    //       // offsetted;
    //       auto begin = non_loc.spans[i].begin;
    //       auto end = non_loc.spans[i].end;
    //       bool local = reparts_to_local(exec_handler, non_loc.rank[i]);
    //       is_local.push_back(local);
    //       if (local) {
    //           // TODO depending on the interface simple non
    //           // transforming and interpolating interfaces like now
    //           // local processor interfaces could be merged
    //           loc.begin.push_back(loc.row_idxs.size());
    //           loc.row_idxs.insert(loc.row_idxs.end(),
    //           non_loc.row_idxs.get_data() + begin,
    //                           non_loc.row_idxs.get_data() + end);
    //           loc.col_idxs.insert(loc.cols.end(), non_loc.col_idxs.get_data()
    //           + begin,
    //                           non_loc.col_idxs.get_data() + end);
    //           loc.ldu_mapping.insert(loc.ldu_mapping.end(),
    //                              non_loc.ldu_mapping.get_data() + begin,
    //                              non_loc.ldu_mapping.get_data() + end);
    //           // TODO store from rank ie from which rank it came
    //           // this is currently unused
    //           // TODO reimplement end doesnt exist
    //           // loc.end.push_back(loc.row_idxs.size());
    //           // also cannot pushback
    //           // loc.rank.push_back(non_loc.rank[i]);
    //       } else {
    //           begins.push_back(row_idxs.size());
    //           rows.insert(rows.end(), non_loc.row_idxs.begin() + begin,
    //                       non_loc.row_idxs.begin() + end);
    //           cols.insert(cols.end(), non_loc.col_idxs.begin() + begin,
    //                       non_loc.col_idxs.begin() + end);
    //           // NOTE we don't do anything with non local mapping
    //           ldu_mapping.insert(ldu_mapping.end(), non_loc.col_idxs.begin()
    //           + begin,
    //                              non_loc.col_idxs.begin() + end);
    //           ends.push_back(rows.size());
    //           ranks.push_back(get_owner_rank(non_loc.rank[i]));
    //       }
    //   }
    //   LOG_1(verbose_, "done build non local interfaces")
    //   return std::make_pair(
    //       SparsityPattern{rows, cols, ldu_mapping, begins, ends, ranks},
    //       is_local);
}
