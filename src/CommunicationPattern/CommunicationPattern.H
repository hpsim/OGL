// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once
#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"

#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"

// TODO move this file to top level since it is not really a matrix-wrapper

using CommCounts = std::tuple<std::vector<label>, std::vector<label>,
                              std::vector<label>, std::vector<label>>;

CommCounts invert_comm_pattern(CommCounts in_pattern);

CommCounts compute_scatter_counts(const ExecutorHandler &exec_handler,
                                  label ranks_per_gpu, label size);

// TODO rename function to compute_gather_to_owner_comm
/* @brief  This function computes the send and recv counts vectors for
 *repartitioning to an owner rank
 *
 * @parameter size how many elements to send to owner
 * @parameter total_size size of the block of memory on the receiver site
 * @parameter padding_before unused elements before inserting size elements
 * @parameter padding_after unused elements after insesting size elements
 * */
CommCounts compute_send_recv_counts(const ExecutorHandler &exec_handler,
                                    label ranks_per_gpu, label size,
                                    label total_size, label padding_before,
                                    label padding_after);

/* @brief  This function computes the send and recv counts vectors for
 *repartitioning to an owner rank
 *
 * @parameter size how many elements to send to owner
 * */
CommCounts compute_send_recv_counts(const ExecutorHandler &exec_handler,
                                    label ranks_per_gpu, label size);

/* @brief  This function computes the send and recv counts vectors for
 *repartitioning to an owner rank
 *
 * @parameter size how many elements to send to owner
 * */
CommCounts compute_scatter_from_owner_comm(const ExecutorHandler &exec_handler,
                                           label ranks_per_gpu, label size);

/* Given a rank id and ranks_per_gpus this function computes the corresponding
 * owner rank  */
label compute_owner_rank(label rank, label ranks_per_gpu);

// TODO these are separate functions, the scalar version generally operates
// on the device where as labels are only communicated on the host side
// create gather_coeffs, gather_idx, and a templated impl function
/* Given a comm_pattern this function executes the communication
**
*/
void communicate_values(const ExecutorHandler &exec_handler,
                        const CommCounts &comm_pattern,
                        const scalar *send_buffer, scalar *recv_buffer);

/* Returns a vector of length total received elements + padding */
std::vector<label> gather_to_owner(const ExecutorHandler &exec_handler,
                                   const CommCounts &comm_pattern, label size,
                                   const label *data, label offset = 0);

struct CommunicationPattern {
    using comm_size_type = label;

    const ExecutorHandler &exec_handler;

    // an array storing to which rank to communicate
    gko::array<comm_size_type> target_ids;

    // an array storing how many elements to communicate
    // to the corresponding target_id
    gko::array<comm_size_type> target_sizes;

    // send_idx stores the index_set ie which cells
    // are owned by the interface and the corresponding target rank
    std::vector<std::pair<gko::array<label>, comm_size_type>> send_idxs;

    CommunicationPattern(
        const ExecutorHandler &exec, gko::array<comm_size_type> ids,
        gko::array<comm_size_type> sizes,
        std::vector<std::pair<gko::array<label>, comm_size_type>> idxs)
        : exec_handler(exec),
          target_ids(ids),
          target_sizes(sizes),
          send_idxs(idxs)
    {
        ASSERT_EQ(target_ids.get_size(), target_sizes.get_size());
        // ASSERT_EQ(target_ids.get_size(), send_idxs.size());
    };

    const gko::experimental::mpi::communicator &get_comm() const
    {
        return *exec_handler.get_communicator().get();
    }


    gko::array<label> total_rank_send_idx() const
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


    CommCounts send_recv_pattern() const
    {
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

        return CommCounts(send_offsets, send_counts, recv_offsets, recv_counts);
    }
};

std::ostream &operator<<(std::ostream &out, const CommunicationPattern &e);
