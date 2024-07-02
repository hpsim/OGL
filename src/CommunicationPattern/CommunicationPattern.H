// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"

#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"

using CommCounts = std::tuple<std::vector<label>, std::vector<label>,
                              std::vector<label>, std::vector<label>>;

/* @brief  This function computes the send and recv counts vectors for
 * scattering from an owner to non  owner ranks
 *
 * @param exec_handler The executor handler
 * @param ranks_per_owner Ratio of the total number of ranks to owner ranks
 * @param size how many elements to send to owner
 * */
CommCounts compute_scatter_from_owner_counts(
    const ExecutorHandler &exec_handler, label ranks_per_owner, label size);

/* @brief  This function computes the send and recv counts and offset vectors
 *for repartitioning to an owner rank
 *
 * @param exec_handler The executor handler
 * @param ranks_per_owner Ratio of the total number of ranks to owner ranks
 * @param size how many elements to send to owner
 * @param total_size Size of the total buffer
 * @param padding_before Number of unused elements before
 * @param padding_after Number of unused elements after
 * @return The CommCounts
 * */
CommCounts compute_gather_to_owner_counts(const ExecutorHandler &exec_handler,
                                          label ranks_per_owner, label size,
                                          label total_size,
                                          label padding_before,
                                          label padding_after);

/* @brief  This function computes the send and recv counts and offset vectors
 *for repartitioning to an owner rank
 *
 * @param exec_handler The executor handler
 * @param ranks_per_owner Ratio of total ranks to owner ranks
 * @parameter size how many elements to send to owner
 * @return The CommCounts
 * */
CommCounts compute_gather_to_owner_counts(const ExecutorHandler &exec_handler,
                                          label ranks_per_owner, label size);

/** @brief Given a rank id and ranks_per_owner this function computes the
 * corresponding owner rank
 *
 * @param rank The rank under consideration
 * @param ranks_per_owner Ratio of total ranks to owner ranks
 * @return The corresponding owner of this rank
 * */
label compute_owner_rank(label rank, label ranks_per_owner);

/* @brief Given a comm_pattern this function executes the communication
 *
 * The purpose of this function is to have a uniform function for communication
 * calls so that we can inspect the communication for debugging purposes
 *
 * @param exec_handler The executor handler
 * @param comm_pattern The communication pattern i.e. the send, recv counts and
 * offsets
 * @param send_buffer Pointer to the data buffer to send
 * @param recv_buffer Pointer to the data buffer to recv
 */
void communicate_values(const ExecutorHandler &exec_handler,
                        const CommCounts &comm_pattern,
                        const scalar *send_buffer, scalar *recv_buffer);

/* @brief A function that can perform an offseted gather to owner rank
 *
 * This function is similar to communicate_values. However, it takes care of
 * allocating the revc_buffer on the owner rank and can add a prescribed offset
 * to all elements. The latter is often required when gathering indices to the
 * owner rank.
 *
 * @param exec_handler The executor handler
 * @param send_buffer Pointer to send data
 * @param send_size How many elements to send to owner
 * @param offset Value to add to all elements in the send_buffer
 * Returns a vector of labels with a length equal the total received elements +
 * padding */
std::vector<label> gather_labels_to_owner(const ExecutorHandler &exec_handler,
                                          const CommCounts &comm_pattern,
                                          const label *send_buffer,
                                          label send_size, label offset = 0);

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
