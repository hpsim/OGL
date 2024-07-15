// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/CommunicationPattern.H"
#include "OGL/common.H"


label compute_owner_rank(label rank, label ranks_per_owner)
{
    return rank - (rank % ranks_per_owner);
}

AllToAllPattern compute_gather_to_owner_counts(const ExecutorHandler &exec_handler,
                                    label ranks_per_owner, label size)
{
    return compute_gather_to_owner_counts(exec_handler, ranks_per_owner, size, size, 0,
                                    0);
}

AllToAllPattern compute_scatter_from_owner_counts(const ExecutorHandler &exec_handler,
                                  label ranks_per_owner, label size)
{
    auto exec = exec_handler.get_device_exec();
    auto comm = *exec_handler.get_communicator().get();

    label total_ranks{comm.size()};
    label rank{comm.rank()};
    label owner_rank = compute_owner_rank(rank, ranks_per_owner);
    std::vector<label> send_counts(total_ranks, 0);
    std::vector<label> recv_counts(total_ranks, 0);
    std::vector<label> send_offsets(total_ranks + 1, 0);
    std::vector<label> recv_offsets(total_ranks + 1, 0);

    label comm_elements_buffer{0};
    if (rank == owner_rank) {
        // send and recv to it self
        recv_offsets[owner_rank] = 0;
        send_counts[owner_rank] = size;
        recv_counts[owner_rank] = size;
        // the start of the next rank data
	label tot_send_elements{size};

        for (int i = 1; i < ranks_per_owner; i++) {
            // receive the recv counts
            comm.recv(exec, &comm_elements_buffer, 1, rank + i, rank);
            send_counts[rank + i] = comm_elements_buffer;
            send_offsets[rank + i] = tot_send_elements;
            tot_send_elements += comm_elements_buffer;
        }
	send_offsets.back() = tot_send_elements;
    } else {
        // send how many elements to communicate
        comm.send(exec, &size, 1, owner_rank, owner_rank);
        recv_counts[owner_rank] = size;
    }

    // the total amount of received elements should be
    // always size
    recv_offsets.back() = size;

    return AllToAllPattern{
        send_counts,
            send_offsets,
            recv_counts,
            recv_offsets
    };
}

AllToAllPattern compute_gather_to_owner_counts(const ExecutorHandler &exec_handler,
                                    label ranks_per_owner, label size,
                                    label total_size, label padding_before,
                                    label padding_after)
{
    auto exec = exec_handler.get_device_exec();
    auto comm = *exec_handler.get_communicator().get();

    ASSERT_EQ(total_size, size + padding_before + padding_after);
    label total_ranks{comm.size()};
    label rank{comm.rank()};
    label owner_rank = compute_owner_rank(rank, ranks_per_owner);
    std::vector<label> send_counts(total_ranks, 0);
    std::vector<label> recv_counts(total_ranks, 0);
    // last entry of offsets vector for total sum
    std::vector<label> send_offsets(total_ranks + 1, 0);
    std::vector<label> recv_offsets(total_ranks + 1, 0);

    label tot_recv_elements{0};
    label comm_elements_buffer{0};
    if (rank == owner_rank) {
        // send and recv to it self
        recv_offsets[owner_rank] = padding_before;
        send_counts[owner_rank] = size;
        recv_counts[owner_rank] = size;
        // the start of the next rank data
        tot_recv_elements = padding_before + size + padding_after;

        for (int i = 1; i < ranks_per_owner; i++) {
            // receive the recv counts
            comm.recv(exec, &comm_elements_buffer, 1, rank + i, rank);
            recv_counts[rank + i] = comm_elements_buffer;
            label recv_size = comm_elements_buffer;

            // receive the padding before
            comm.recv(exec, &comm_elements_buffer, 1, rank + i, rank);
            tot_recv_elements += comm_elements_buffer;
            recv_offsets[rank + i] = tot_recv_elements;

            // receive the padding after and keep for next rank
            // as new offset
            tot_recv_elements += recv_size;
            comm.recv(exec, &comm_elements_buffer, 1, rank + i, rank);
            tot_recv_elements += comm_elements_buffer;
        }
    } else {
        // send how many elements to communicate
        comm.send(exec, &size, 1, owner_rank, owner_rank);
        send_counts[owner_rank] = size;

        // send how much padding before is needed
        comm.send(exec, &padding_before, 1, owner_rank, owner_rank);

        // send how much padding after is needed
        comm.send(exec, &padding_after, 1, owner_rank, owner_rank);
    }

    send_offsets[total_ranks] =
        std::accumulate(send_counts.begin(), send_counts.end(), 0);
    recv_offsets[total_ranks] =
        std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    return AllToAllPattern{
        send_counts, send_offsets,
            recv_counts,recv_offsets};
}


void communicate_values(const ExecutorHandler &exec_handler,
                        const AllToAllPattern &comm_pattern,
                        const scalar *send_buffer, scalar *recv_buffer)
{
    auto exec = exec_handler.get_device_exec();
    auto comm = *exec_handler.get_communicator().get();

    // TODO add some sanity checks for comm pattern
    // 1. length needs to be same as mpi ranks
    // 2. recv_buffer length should match ie at least length of recv_counts

    // std:::cout
    //     << __FILE__ << ":" << __LINE__
    //     << " send_counts " <<   send_counts
    //     << " recv_counts " << recv_counts
    //     << " send_offsets " << send_offsets
    //     << " recv_offsets " << recv_offsets
    //     << "\n";

    comm.all_to_all_v(exec, send_buffer, comm_pattern.send_counts.data(),
                      comm_pattern.send_offsets.data(), recv_buffer, comm_pattern.recv_counts.data(),
                      comm_pattern.recv_offsets.data());
}

std::vector<label> gather_labels_to_owner(const ExecutorHandler &exec_handler,
                                   const AllToAllPattern &comm_pattern,
                                   const label *send_buffer,
                                   label send_size,
                                   label offset)
{
    std::vector<label> send_buffer_copy;
    // create a copy if offset is needed
    if (offset > 0) {
        send_buffer_copy.resize(send_size);
        std::transform(send_buffer, send_buffer + send_size, send_buffer_copy.data(),
                       [&](label idx) { return idx + offset; });
    }

    auto &[send_counts, recv_counts, send_offsets, recv_offsets] = comm_pattern;
    auto total_recv_size = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    auto exec = exec_handler.get_ref_exec();
    auto comm = *exec_handler.get_communicator().get();
    std::vector<label> recv_buffer(total_recv_size);
    auto rank = comm.rank();
    comm.all_to_all_v(exec,
            (offset > 0) ? send_buffer_copy.data() : send_buffer,
            send_counts.data(),
                      send_offsets.data(), recv_buffer.data(),
                      recv_counts.data(), recv_offsets.data());
    return recv_buffer;
}

std::ostream &operator<<(std::ostream &out, const CommunicationPattern &e)
{
    // TODO add implementation
    // out << "CommunicationPattern: for rank: " << e.get_comm().rank();
    // out << " {";
    // out << "\ntarget_ids: " << e.target_ids;
    // out << "\ntarget_sizes: " << e.target_sizes;
    // out << "}\n";
    return out;
}


gko::array<label> CommunicationPattern::total_rank_send_idx() const
{
    std::vector<label> tmp;

    for (auto &[arr, id] : send_idxs) {
        label arr_size = arr.get_size();
        tmp.insert(tmp.end(), arr.get_const_data(),
                   arr.get_const_data() + arr_size);
    }

    return gko::array<label>(exec_handler.get_ref_exec(), tmp.begin(),
                             tmp.end());
}


AllToAllPattern CommunicationPattern::send_recv_pattern() const {
        auto comm = *exec_handler.get_communicator().get();
        label total_ranks{comm.size()};

        std::vector<label> send_counts(comm.size());
        std::vector<label> send_offsets(comm.size() + 1);
        std::vector<label> recv_counts(comm.size());
        std::vector<label> recv_offsets(comm.size() + 1);

        label comm_ranks = target_ids.get_size();
        label tot_comm_size = 0;
        for (label i = 0; i < comm_ranks; i++) {
            auto comm_rank = target_ids.get_const_data()[i];
            auto comm_size = target_sizes.get_const_data()[i];
            tot_comm_size += comm_size;
            send_counts[comm_rank] = comm_size;
            recv_counts[comm_rank] = comm_size;
        }

        recv_offsets[comm.size()] = tot_comm_size;
        std::partial_sum(recv_counts.begin(), recv_counts.end(),
                         recv_offsets.begin() + 1);
        recv_offsets[0] = 0;

        std::partial_sum(send_counts.begin(), send_counts.end(),
                         send_offsets.begin() + 1);
        send_offsets[0] = 0;

        return AllToAllPattern{send_counts, send_offsets, recv_counts,
                               recv_offsets};
}
