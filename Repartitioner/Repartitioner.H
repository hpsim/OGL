/*---------------------------------------------------------------------------* \
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
#pragma once

#include <ginkgo/ginkgo.hpp>

#include "MatrixWrapper/CommunicationPattern/CommunicationPattern.H"
#include "MatrixWrapper/LDUMatrix/HostMatrix.H"

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

// TODO this should be persistent
class Repartitioner {
private:
    using partition = gko::experimental::distributed::Partition<label, label>;

    const label size_;  //! Size (n matrix rows ) before repartitioning

    const label repart_size_;  //! Size after repartitioning

    const label ranks_per_gpu_;

    std::shared_ptr<const partition> orig_partition_;


public:
    Repartitioner(label size, label ranks_per_gpu,
                  const ExecutorHandler &exec_handler)
        : size_(size),
          repart_size_(Repartitioner::compute_repart_size(size, ranks_per_gpu,
                                                          exec_handler)),
          ranks_per_gpu_(ranks_per_gpu),
          orig_partition_(gko::share(
              gko::experimental::distributed::build_partition_from_local_size<
                  label, label>(exec_handler.get_device_exec(),
                                *exec_handler.get_communicator().get(),
                                size))) {};

    /* returns the owner rank for a given rank */
    label get_owner_rank(label rank) const
    {
        return compute_owner_rank(rank, ranks_per_gpu_);
    };

    /* returns the owner rank for a given rank */
    label get_owner_rank(
 const ExecutorHandler& exec_handler
) const { return get_owner_rank(get_rank(exec_handler)); };

    /* returns if current rank is an owner  */
    bool is_owner(
 const ExecutorHandler& exec_handler
) const { return get_rank(exec_handler) == get_owner_rank(get_rank(exec_handler)); };

    /* @brief check if the given rank gets local after repartitioning
     *
     * */
    bool reparts_to_local(
 const ExecutorHandler& exec_handler,
        label rank) const
    {
        return get_owner_rank(exec_handler) == compute_owner_rank(rank, ranks_per_gpu_);
    };

    /* shortcut to current rank */
    label get_rank(const ExecutorHandler& exec_handler) const
    {
        return exec_handler.get_communicator().get()->rank();
    };

    label get_ranks_per_gpu() const { return ranks_per_gpu_; }

    static label compute_repart_size(label local_size, label ranks_per_gpu,
                                     const ExecutorHandler &exec_handler)
    {
        if (ranks_per_gpu == 1) {
            return local_size;
        }

        CommCounts comm_pattern =
            compute_send_recv_counts(exec_handler, ranks_per_gpu, 1);

        std::vector<label> data{local_size};

        auto local_sizes =
            gather_to_owner(exec_handler, comm_pattern, 1, data.data());

        return std::accumulate(local_sizes.begin(), local_sizes.end(), 0);
    }

    label get_repart_size() const { return repart_size_; }

    gko::dim<2> get_repart_dim() const
    {
        return gko::dim<2>{repart_size_, repart_size_};
    }

    std::shared_ptr<const partition> get_orig_partition() const
    {
        return orig_partition_;
    };


    /* @brief given received interfaces this function sorts into local and non
     *local
     * returns: SparsityPatternVector storing new non_local sparsity pattern,
     *  and a pair of locality information, where the first bool stores if the
     *  interface performs a local communication and the second bool if the
     *  the interface was originally from this rank
     */
    std::pair<SparsityPatternVector, std::vector<bool>>
    build_non_local_interfaces(
        const ExecutorHandler& exec_handler,
        SparsityPatternVector &loc,
                               const SparsityPatternVector &non_loc) const
    {
        std::vector<label> rows, cols, ldu_mapping, ranks, begins, ends;
        std::vector<bool> is_local;
        label merged_ranks_size = non_loc.ranks.size();

        for (label i = 0; i < merged_ranks_size; i++) {
            // these are the begin ends before merging they need to be
            // offseted;
            auto begin = non_loc.begin[i];
            auto end = non_loc.end[i];
            bool local = reparts_to_local(exec_handler, non_loc.ranks[i]);
            is_local.push_back(local);
            if (local) {
                // TODO depending on the interface simple non
                // transforming and interpolating interfaces like now
                // local processor interfaces could be merged
                loc.begin.push_back(loc.rows.size());
                loc.rows.insert(loc.rows.end(), non_loc.rows.begin() + begin,
                                non_loc.rows.begin() + end);
                loc.cols.insert(loc.cols.end(), non_loc.cols.begin() + begin,
                                non_loc.cols.begin() + end);
                loc.mapping.insert(loc.mapping.end(),
                                   non_loc.mapping.data() + begin,
                                   non_loc.mapping.data() + end);
                // TODO store from rank ie from which rank it came
                // this is currently unused
                loc.end.push_back(loc.rows.size());
                loc.ranks.push_back(non_loc.ranks[i]);
            } else {
                begins.push_back(rows.size());
                rows.insert(rows.end(), non_loc.rows.begin() + begin,
                            non_loc.rows.begin() + end);
                cols.insert(cols.end(), non_loc.cols.begin() + begin,
                            non_loc.cols.begin() + end);
                // NOTE we don't do anything with non local mapping
                ldu_mapping.insert(ldu_mapping.end(),
                                   non_loc.cols.begin() + begin,
                                   non_loc.cols.begin() + end);
                ends.push_back(rows.size());
                ranks.push_back(get_owner_rank(non_loc.ranks[i]));
            }
        }
        return std::make_pair(
            SparsityPatternVector{rows, cols, ldu_mapping, begins, ends, ranks},
            is_local);
    }


    /*
    ** returns the the new sparsity patterns (local and non-local) and a vector
    ** tracking the interfaces. This contains a pair where the first entry
    *(bool)
    ** signals whether this is a new local interface (no communication), and the
    ** second entry (label) tracks the original rank of the interface
    */
    std::tuple<std::shared_ptr<SparsityPattern>,
               std::shared_ptr<SparsityPattern>,
               std::vector<std::pair<bool, label>>>
    repartition_sparsity(
        const ExecutorHandler& exec_handler,
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
        auto exec = exec_handler.get_device_exec();
        auto comm = *exec_handler.get_communicator().get();
        label rank = get_rank(exec_handler);
        label owner_rank = get_owner_rank(exec_handler);
        label ranks_per_gpu = ranks_per_gpu_;

        auto local_comm_pattern = compute_send_recv_counts(
            exec_handler, ranks_per_gpu, src_local_pattern->size_);

        label offset = orig_partition_->get_range_bounds()[rank] -
                       orig_partition_->get_range_bounds()[owner_rank];

        auto gather_closure = [&](auto &comm_pattern, auto &data,
                                  label offset) {
            return gather_to_owner(exec_handler, comm_pattern, data.get_size(),
                                   data.get_data(), offset);
        };

        SparsityPatternVector merged_local{
            gather_closure(local_comm_pattern, src_local_pattern->row_idxs,
                           offset),
            gather_closure(local_comm_pattern, src_local_pattern->col_idxs,
                           offset),
            gather_closure(local_comm_pattern, src_local_pattern->ldu_mapping,
                           0),
            std::vector<label>{0},
            std::vector<label>{0},
            std::vector{rank}};
        merged_local.end[0] = merged_local.rows.size();

        if (is_owner(exec_handler)) {
            make_ldu_mapping_consecutive(
                local_comm_pattern, merged_local.mapping, rank, ranks_per_gpu);
        }

        label rows = (is_owner(exec_handler)) ? merged_local.rows.back() + 1 : 0;
        gko::dim<2> merged_local_dim{rows, rows};

        auto non_local_comm_pattern = compute_send_recv_counts(
            exec_handler, ranks_per_gpu, src_non_local_pattern->size_);

        std::vector<label> spans_begin;
        std::vector<label> spans_end;
        auto span_comm_pattern = compute_send_recv_counts(
            exec_handler, ranks_per_gpu,
            src_non_local_pattern->interface_spans.size());
        for (auto elem : src_non_local_pattern->interface_spans) {
            spans_begin.push_back(elem.begin);
            spans_end.push_back(elem.end);
        }

        // the non local cols are in local idx of other side
        // thus we need the new offset of the other side
        // NOTE TODO this modifies src_non_local_pattern better do the offsets
        // on the owner side after gathering
        for (label i = 0; i < src_non_local_pattern->rank.size(); i++) {
            auto comm_rank = src_non_local_pattern->rank[i];
            auto [begin, end] = src_non_local_pattern->interface_spans[i];
            label local_offset =
                orig_partition_->get_range_bounds()[comm_rank] -
                orig_partition_->get_range_bounds()[owner_rank];
            auto *data = src_non_local_pattern->col_idxs.get_data() + begin;
            auto size = end - begin;
            std::transform(data, data + size, data,
                           [&](label idx) { return idx + local_offset; });
        }

        SparsityPatternVector merged_non_local{
            gather_closure(non_local_comm_pattern,
                           src_non_local_pattern->row_idxs, offset),
            gather_closure(non_local_comm_pattern,
                           src_non_local_pattern->col_idxs, 0),
            gather_closure(non_local_comm_pattern,
                           src_non_local_pattern->row_idxs, 0),
            gather_to_owner(exec_handler, span_comm_pattern, spans_begin.size(),
                            spans_begin.data()),
            gather_to_owner(exec_handler, span_comm_pattern, spans_end.size(),
                            spans_end.data()),
            gather_to_owner(exec_handler, span_comm_pattern,
                            src_non_local_pattern->rank.size(),
                            src_non_local_pattern->rank.data())};

        if (is_owner(exec_handler)) {
            make_begin_and_ends_consecutive(
                span_comm_pattern, merged_non_local.begin, merged_non_local.end,
                rank, ranks_per_gpu);
        }

        auto [gathered_non_local, is_local] =
            build_non_local_interfaces(exec_handler, merged_local, merged_non_local);

        // build vector with locality information
        std::vector<std::pair<bool, label>> locality;
        label ctr{0};

        if (is_owner(exec_handler)) {
            auto recv_counts = std::get<1>(span_comm_pattern);
            for (int i = 0; i < ranks_per_gpu; i++) {
                label comm_rank = rank + i;
                label num_elems = recv_counts[comm_rank];
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
        // keep an interface together thus we can just count the idx up to the size.
        // But we have to make sure that the interfaces are in the same order
        // on both communication sides.
        for (int i = 0; i < gathered_non_local.cols.size(); i++) {
            gathered_non_local.cols[i] = i;
        }

    LOG_1(verbose_, "done repartition sparsity pattern")
        if (is_owner(exec_handler)) {
            auto new_local_spars_pattern = std::make_shared<SparsityPattern>(
                exec, merged_local_dim, merged_local);

            auto new_non_local_spars_pattern =
                std::make_shared<SparsityPattern>(
                    exec,
                    gko::dim<2>{merged_local_dim[0],
                                gathered_non_local.rows.size()},
                    gathered_non_local);

            return std::make_tuple<std::shared_ptr<SparsityPattern>,
                                   std::shared_ptr<SparsityPattern>,
                                   std::vector<std::pair<bool, label>>>(
                std::move(new_local_spars_pattern),
                std::move(new_non_local_spars_pattern), std::move(locality));
        } else {
            auto new_local_spars_pattern =
                std::make_shared<SparsityPattern>(exec);

            auto new_non_local_spars_pattern =
                std::make_shared<SparsityPattern>(exec);

            new_local_spars_pattern->dim = merged_local_dim;
            new_non_local_spars_pattern->dim = merged_local_dim;

            return std::make_tuple<std::shared_ptr<SparsityPattern>,
                                   std::shared_ptr<SparsityPattern>,
                                   std::vector<std::pair<bool, label>>>(
                std::move(new_local_spars_pattern),
                std::move(new_non_local_spars_pattern),
                std::vector<std::pair<bool, label>>{});
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

    std::shared_ptr<const CommunicationPattern> repartition_comm_pattern(
        const ExecutorHandler& exec_handler,
        std::shared_ptr<const CommunicationPattern> src_comm_pattern,
        std::shared_ptr<
            const gko::experimental::distributed::Partition<label, label>>
            partition) const
    {
        using comm_size_type = label;

        if (ranks_per_gpu_ == 1) {
            return src_comm_pattern;
        }

        auto exec = src_comm_pattern->target_ids.get_executor();
        auto comm = src_comm_pattern->get_comm();

        // Step 1. Check if communication partner is non-local after
        // repartitioning. If it is non-local we keep it. Otherwise local
        // communication can be discarded. Here non-local means:
        // the communication target proc id is != repartition proc id
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
        auto comm_pattern = compute_send_recv_counts(
            exec_handler, ranks_per_gpu_, target_ids.size());

        auto gathered_target_ids = gather_to_owner(
            exec_handler, comm_pattern, target_ids.size(), target_ids.data());

        auto gathered_target_sizes =
            gather_to_owner(exec_handler, comm_pattern, target_sizes.size(),
                            target_sizes.data());

        // next the send_ixs need to be updated we send them piecewise since
        // the send_idxs are a vector of gko::arrays
        label rank = get_rank(exec_handler);
        if (is_owner(exec_handler)) {
            auto recv_counts = std::get<1>(comm_pattern);
            label owner_recv_counts = recv_counts[rank];
            // retrieved from i-th neighbor
            for (int i = 1; i < ranks_per_gpu_; i++) {
                // how many gko::arrays to with send_indexes to receive
                // from i-th neighbor
                label recv_count = recv_counts[rank + i];
                for (int j = 0; j < recv_count; j++) {
                    auto target_size =
                        gathered_target_sizes[j + owner_recv_counts];
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
            auto send_count = std::get<0>(comm_pattern)[owner];
            for (int i = 0; i < send_count; i++) {
                auto send_buffer = send_idxs[i].first;
                comm.send(exec, send_buffer.get_const_data(),
                          send_buffer.get_size(), owner, owner);
            }
        }

        // clear communcation neighbors on non owning rank
        if (!is_owner(exec_handler)) {
            // TODO NOTE should it be gathered_target_ids and is it needed?
            target_ids.clear();
            target_sizes.clear();
            send_idxs.clear();
        }

        // early return if no communication partners are left
        if (target_sizes.size() == 0) {
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
            // of elemts and the send_ixs
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
};
