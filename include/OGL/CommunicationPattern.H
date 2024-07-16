// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"

#include "OGL/DevicePersistent/ExecutorHandler.H"

/** Struct holding vectors needded for MPI all to all communication
 */
struct AllToAllPattern {
    std::vector<label> send_counts;
    std::vector<label> send_offsets;
    std::vector<label> recv_counts;
    std::vector<label> recv_offsets;
};

/* @brief  This function computes the send and recv counts vectors and the send
 * and recv offsets vectors for scattering from an owner to all ranks, including
 * owner itself
 *
 * @param exec_handler The executor handler
 * @param ranks_per_owner Ratio of the total number of ranks to owner ranks
 * @param size how many elements to send from the owner back to neighbour ranks
 * */
AllToAllPattern compute_scatter_from_owner_counts(
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
 * @return The AllToAllPattern
 * */
AllToAllPattern compute_gather_to_owner_counts(
    const ExecutorHandler &exec_handler, label ranks_per_owner, label size,
    label total_size, label padding_before, label padding_after);

/* @brief  This function computes the send and recv counts and offset vectors
 *for repartitioning to an owner rank
 *
 * @param exec_handler The executor handler
 * @param ranks_per_owner Ratio of total ranks to owner ranks
 * @parameter size how many elements to send to owner
 * @return The AllToAllPattern
 * */
AllToAllPattern compute_gather_to_owner_counts(
    const ExecutorHandler &exec_handler, label ranks_per_owner, label size);

/** @brief Given a rank id and ranks_per_owner this function computes the
 * corresponding owner rank
 *
 * @param rank The rank under consideration
 * @param ranks_per_owner Ratio of the total number of ranks to owner ranks
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
                        const AllToAllPattern &comm_pattern,
                        const scalar *send_buffer, scalar *recv_buffer);

/* @brief A function that can perform an offsetted gather to owner rank
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
                                          const AllToAllPattern &comm_pattern,
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

    // send_idxs stores the index_set ie which cells
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
        // TODO check size of send_idxs
        //
        // ASSERT_EQ(target_ids.get_size(), send_idxs.size());
    }

    const gko::experimental::mpi::communicator &get_comm() const
    {
        return *exec_handler.get_communicator().get();
    }


    /* @brief concatenate all separate send idxs arrays into one contiguous
     * array
     */
    gko::array<label> total_rank_send_idx() const;


    /** Computes the send and recv pattern for ginkgos distributed matrix SpMV
     */
    AllToAllPattern send_recv_pattern() const;
};

std::ostream &operator<<(std::ostream &out, const CommunicationPattern &e);
