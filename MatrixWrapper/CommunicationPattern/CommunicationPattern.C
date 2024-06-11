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


label compute_owner_rank(label rank, label ranks_per_gpu)
{
    return rank - (rank % ranks_per_gpu);
};

CommCounts
compute_send_recv_counts(const ExecutorHandler &exec_handler,
                         label ranks_per_gpu, label size)
{
    return compute_send_recv_counts(exec_handler, ranks_per_gpu, size, size, 0,
                                    0);
}

CommCounts
compute_scatter_counts(const ExecutorHandler &exec_handler,
                         label ranks_per_gpu, label size)
{
    auto exec = exec_handler.get_device_exec();
    auto comm = *exec_handler.get_communicator().get();

    label total_ranks{comm.size()};
    label rank{comm.rank()};
    label owner_rank = compute_owner_rank(rank, ranks_per_gpu);
    std::vector<label> send_counts(total_ranks, 0);
    std::vector<label> recv_counts(total_ranks, 0);
    std::vector<label> send_offsets(total_ranks, 0);
    std::vector<label> recv_offsets(total_ranks, 0);

    label tot_recv_elements{0};
    label comm_elements_buffer{0};
    if (rank == owner_rank) {
        // send and recv to it self
        recv_offsets[owner_rank] = 0;
        send_counts[owner_rank] = size;
        recv_counts[owner_rank] = size;
        // the start of the next rank data
        tot_recv_elements = size;

        for (int i = 1; i < ranks_per_gpu; i++) {
            // receive the recv counts
            comm.recv(exec, &comm_elements_buffer, 1, rank + i, rank);
            send_counts[rank + i] = comm_elements_buffer;
            send_offsets[rank + i] = tot_recv_elements;
            tot_recv_elements += comm_elements_buffer;
        }
    } else {
        // send how many elements to communicate
        comm.send(exec, &size, 1, owner_rank, owner_rank);
        recv_counts[owner_rank] = size;
    }

    return std::make_tuple(send_counts, recv_counts, send_offsets,
                           recv_offsets);
}

CommCounts
compute_send_recv_counts(const ExecutorHandler &exec_handler,
                         label ranks_per_gpu, label size, label total_size,
                         label padding_before, label padding_after)
{
    auto exec = exec_handler.get_device_exec();
    auto comm = *exec_handler.get_communicator().get();

    ASSERT_EQ(total_size, size + padding_before + padding_after);
    label total_ranks{comm.size()};
    label rank{comm.rank()};
    label owner_rank = compute_owner_rank(rank, ranks_per_gpu);
    std::vector<label> send_counts(total_ranks, 0);
    std::vector<label> recv_counts(total_ranks, 0);
    std::vector<label> send_offsets(total_ranks, 0);
    std::vector<label> recv_offsets(total_ranks, 0);

    label tot_recv_elements{0};
    label comm_elements_buffer{0};
    std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank << "\n";
    if (rank == owner_rank) {
        // send and recv to it self
        recv_offsets[owner_rank] = padding_before;
        send_counts[owner_rank] = size;
        recv_counts[owner_rank] = size;
        // the start of the next rank data
        tot_recv_elements = padding_before + size + padding_after;

	std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank << "\n";
        for (int i = 1; i < ranks_per_gpu; i++) {
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

    return std::make_tuple(send_counts, recv_counts, send_offsets,
                           recv_offsets);
}


void communicate_values (
    const ExecutorHandler &exec_handler,
    const CommCounts &comm_pattern,
    const scalar *send_buffer, scalar *recv_buffer)
{
    auto exec = exec_handler.get_device_exec();
    auto comm = *exec_handler.get_communicator().get();
    auto [send_counts, recv_counts, send_offsets, recv_offsets] = comm_pattern;
    // TODO add some sanity checks for comm pattern
    // 1. length needs to be same as mpi ranks
    // 2. recv_buffer length should match ie at least length of recv_counts

    // NOTE old holes are captured by comm_pattern
    // the default comm pattern needs to be adjusted
    // since we might also need to leave "holes" between the
    // gathered blocks thus the offset / distance between data
    // should be before repartitioning
    //
    // send_buffer should be on the host
    // recv_buffer should be on the device
    // auto rank = comm.rank();
    std::cout
	    << __FILE__ << ":" << __LINE__
	    << " send_counts " <<   send_counts
	    << " recv_counts " << recv_counts
	    << " send_offsets " << send_offsets
	    << " recv_offsets " << recv_offsets
	    << "\n";

    comm.all_to_all_v(exec, send_buffer, send_counts.data(),
                      send_offsets.data(), recv_buffer, recv_counts.data(),
                      recv_offsets.data());
};

std::vector<label> gather_to_owner(
    const ExecutorHandler &exec_handler,
    const CommCounts &comm_pattern,
    label size, const label *data, label offset)
{
    auto exec = exec_handler.get_ref_exec();
    auto comm = *exec_handler.get_communicator().get();
    std::vector<label> send_buffer_copy(size);
    if (offset > 0) {
        std::transform(data, data + size, send_buffer_copy.data(),
                       [&](label idx) { return idx + offset; });
    }

    const label *send_buffer = (offset > 0) ? send_buffer_copy.data() : data;

    auto &[send_counts, recv_counts, send_offsets, recv_offsets] = comm_pattern;

    // compute total recv buffer size
    // TODO account for holes
    auto total_recv_size = 0;
    for (int i = 0; i < recv_counts.size(); i++) {
        total_recv_size += recv_counts[i];
    }

    std::vector<label> recv_buffer(total_recv_size);
    auto rank = comm.rank();
    comm.all_to_all_v(exec, send_buffer, send_counts.data(),
                      send_offsets.data(), recv_buffer.data(),
                      recv_counts.data(), recv_offsets.data());
    return recv_buffer;
}

std::ostream &operator<<(std::ostream &out, const CommunicationPattern &e)
{
    out << "CommunicationPattern: for rank: " << e.get_comm().rank();
    out << " {";
    out << "\ntarget_ids: " << e.target_ids;
    out << "\ntarget_sizes: " << e.target_sizes;
    out << "}\n";
    return out;
}
