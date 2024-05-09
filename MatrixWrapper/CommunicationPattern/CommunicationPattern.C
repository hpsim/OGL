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

Class
    Foam::IOSortingIdxHandler

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    CommunicationPattern.H

\*---------------------------------------------------------------------------*/

#include "MatrixWrapper/CommunicationPattern/CommunicationPattern.H"
#include "common/common.H"

std::tuple<std::vector<label>, std::vector<label>, std::vector<label>,
           std::vector<label>, std::vector<label>>
compute_send_recv_counts(label ranks_per_gpu, label owner_rank, label size,
                         const gko::experimental::mpi::communicator &comm,
                         std::shared_ptr<const gko::Executor> exec)
{
    label total_ranks{comm.size()};
    label rank{comm.rank()};
    std::vector<label> send_counts(total_ranks, 0);
    std::vector<label> recv_counts(total_ranks, 0);
    std::vector<label> send_offsets(total_ranks, 0);
    std::vector<label> recv_offsets(total_ranks, 0);

    label tot_recv_elements{0};
    label comm_elements_buffer{0};
    if (rank == owner_rank) {
        // send and recv to it self
        send_counts[owner_rank] = size;
        recv_counts[owner_rank] = size;
        tot_recv_elements = size;

        for (int i = 1; i < ranks_per_gpu; i++) {
            comm.recv(exec, &comm_elements_buffer, 1, rank + i, rank);
            recv_offsets[rank + i] = tot_recv_elements;
            recv_counts[rank + i] = comm_elements_buffer;
            tot_recv_elements += comm_elements_buffer;
        }
    } else {
        comm.send(exec, &size, 1, owner_rank, owner_rank);
        send_counts[owner_rank] = size;
    }

    std::vector<label> target_recv_buffer(tot_recv_elements, 0);

    return std::make_tuple(send_counts, recv_counts, send_offsets, recv_offsets,
                           target_recv_buffer);
}

std::ostream &operator<<(std::ostream &out, const CommunicationPattern &e)
{
    out << "CommunicationPattern: for rank: " << e.comm.rank();
    out << " {";
    out << "\ntarget_ids: " << e.target_ids;
    out << "\ntarget_sizes: " << e.target_sizes;
    out << "}\n";
    return out;
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

template <typename T>
std::vector<T> apply_permutation(const std::vector<T> vec,
                                 const std::vector<label> &p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
                   [&](label i) { return vec[i]; });
    return sorted_vec;
}

CommunicationPattern repartition_comm_pattern(
    label ranks_per_gpu, CommunicationPattern &src_comm_pattern,
    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
        partition)
{
    using comm_size_type = label;

    if (ranks_per_gpu == 1) {
        return CommunicationPattern(src_comm_pattern);
    }

    auto exec = src_comm_pattern.target_ids.get_executor();
    auto target_rank = [ranks_per_gpu](label rank) {
        return rank - (rank % ranks_per_gpu);
    };

    auto comm = src_comm_pattern.comm;

    // find ranks which are now local
    // the target_ids, target_sizes etc need to be communicated
    // with new owner. thus we use maps to store them where the index
    // is the new owner.
    std::vector<label> target_ids{};
    std::vector<label> target_sizes{};
    std::vector<std::pair<gko::array<label>, comm_size_type>> send_idxs;
    label rank{src_comm_pattern.comm.rank()};
    label owner_rank = target_rank(rank);
    bool owner = rank == owner_rank;

    // map for exchanging communication data where the index is the target rank
    for (int i = 0; i < src_comm_pattern.target_ids.get_size(); i++) {
        label target_id = src_comm_pattern.target_ids.get_data()[i];
        // Step 1. Check if communication partner is non-local after
        // repartitioning. If it is non-local we keep it. Otherwise local
        // communication can be discarded or merged. Here non-local means: the
        // communication target proc id is != repartition proc id
        //
        // [ 1 | 2 | 3 ] [ 4 | 5 | 6 ] where [] - bounds after repart
        //                                    | - bounds before repart
        // e.g. rank 3 has rank 2 and 4 as comm neighbours
        //      rank 2 can be discarded but rank 4 is non-local and needs
        //      to be communicated to rank 1 to be merged into the new
        //      communicaton pattern. This needs an all to all communication.

        label repart_target_id = target_rank(target_id);
        label target_size = src_comm_pattern.target_sizes.get_const_data()[i];
        auto send_idx = src_comm_pattern.send_idxs[i];

        if (repart_target_id != owner_rank) {
            // communication pattern is non local, hence we keep it
            // after repartitioning we now have to communicate with a different
            // rank
            // store new owner rank to which repart_target_* needs to be send to
            target_ids.push_back(repart_target_id);
            target_sizes.push_back(target_size);
            send_idxs.push_back(send_idx);
        }
    }

    // where to put elements from recv
    // communicate with all neighbours which are to be merged
    // how many elements the owner rank is to receive
    auto [send_counts, recv_counts, send_offsets, recv_offsets,
          target_recv_buffer] =
        compute_send_recv_counts(ranks_per_gpu, owner_rank, target_ids.size(),
                                 comm, exec);


    comm.all_to_all_v(exec, target_ids.data(), send_counts.data(),
                      send_offsets.data(), target_recv_buffer.data(),
                      recv_counts.data(), recv_offsets.data());

    target_ids = target_recv_buffer;

    comm.all_to_all_v(exec, target_sizes.data(), send_counts.data(),
                      send_offsets.data(), target_recv_buffer.data(),
                      recv_counts.data(), recv_offsets.data());

    target_sizes = target_recv_buffer;

    // communicate the send indices to owner rank
    // send indices are packed into std::vector of gko::arrays thus we send them
    // one by one
    // additionally we already have then target_sizes and send_counts and
    // recv_counts which can be reused to for sending the index vectors

    // TODO this could be refactored to separtate function
    // this is also used distributed
    if (owner) {
        // retrieve from all neighbours
        for (int i = 1; i < ranks_per_gpu; i++) {
            // retrieve all send_idxs
            for (int j = 0; j < recv_counts[rank + i]; j++) {
                auto target_size = target_sizes[j + 1];
                auto target_id = target_ids[j + 1];
                std::vector<label> recv_buffer(target_size);

                comm.recv(exec, recv_buffer.data(), target_size, rank + i,
                          rank);

                // the new offset is
                auto offset = partition->get_range_bounds()[rank + i] -
                              partition->get_range_bounds()[rank];

                std::transform(recv_buffer.begin(), recv_buffer.end(),
                               recv_buffer.begin(),
                               [&](label idx) { return idx + offset; });

                send_idxs.push_back({gko::array<label>{
                                         exec,
                                         recv_buffer.begin(),
                                         recv_buffer.end(),
                                     },
                                     target_id});
            }
        }
    } else {
        // send all comm pairs to new owner
        auto send_count = send_counts[owner_rank];
        for (int i = 0; i < send_count; i++) {
            auto send_buffer = send_idxs[i].first;
            comm.send(exec, send_buffer.get_const_data(),
                      send_buffer.get_size(), owner_rank, owner_rank);
        }
    }


    // clear communcation neighbors on non owning rank
    if (!owner) {
        target_ids.clear();
        target_sizes.clear();
        send_idxs.clear();
    }

    // Step 3. merge communication
    // we now might have communication partners multiple times, thus we can
    // merge them sort repart_target_ids and use the sorting for target_sizes
    // and repart_send_idxs
    std::vector<label> merged_target_ids{};
    std::vector<label> merged_target_sizes{};
    std::vector<std::vector<label>> merged_send_idxs{};
    if (owner) {
        auto p = sort_permutation(target_ids,
                                  [](label a, label b) { return a < b; });

        target_sizes = apply_permutation(target_sizes, p);
        target_ids = apply_permutation(target_ids, p);
        send_idxs = apply_permutation(send_idxs, p);

        // Step 4.
        // Merge communication pattern with corresponding neighbours
        if (target_ids.size() > 1) {
            merged_target_ids.push_back(target_ids[0]);
            merged_target_sizes.push_back(target_sizes[0]);
            merged_send_idxs.emplace_back(std::vector<label>(
                send_idxs[0].first.get_data(),
                send_idxs[0].first.get_data() + send_idxs[0].first.get_size()));
        }

        for (int i = 1; i < target_ids.size(); i++) {
            if (target_ids[i] == merged_target_ids.back()) {
                merged_target_sizes.back() += target_sizes[i];
                merged_send_idxs.back().insert(
                    merged_send_idxs.back().end(),
                    send_idxs[i].first.get_data(),
                    send_idxs[i].first.get_data() +
                        send_idxs[i].first.get_size());
            } else {
                merged_target_ids.push_back(target_ids[i]);
            }
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

    return CommunicationPattern{
        src_comm_pattern.comm,
        gko::array<comm_size_type>{exec, merged_target_ids.begin(),
                                   merged_target_ids.end()},
        gko::array<comm_size_type>{exec, merged_target_sizes.begin(),
                                   merged_target_sizes.end()},
        send_idxs};
}
