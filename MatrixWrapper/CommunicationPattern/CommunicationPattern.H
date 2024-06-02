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

#pragma once
#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"

#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"

// TODO move this file to top level since it is not really a matrix-wrapper

using CommCounts = std::tuple<std::vector<label>, std::vector<label>,
                              std::vector<label>, std::vector<label>>;


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
};

std::ostream &operator<<(std::ostream &out, const CommunicationPattern &e);
