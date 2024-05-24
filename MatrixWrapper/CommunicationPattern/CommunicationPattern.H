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

// TODO move this file to top level since it is not really a matrix-wrapper

using CommCounts =
std::tuple<std::vector<label>, std::vector<label>, std::vector<label>,
           std::vector<label>>;


// This function computes the send and recv counts vectors for repartitioning
CommCounts
compute_send_recv_counts(label ranks_per_gpu, label owner_rank,
                         const gko::experimental::mpi::communicator &comm,
                         std::shared_ptr<const gko::Executor> exec, label size,
                         label total_size, label padding_before,
                         label padding_after);

// This function computes the send and recv counts vectors for repartitioning
CommCounts
compute_send_recv_counts(label ranks_per_gpu, label owner_rank,
                         const gko::experimental::mpi::communicator &comm,
                         std::shared_ptr<const gko::Executor> exec, label size);

// TODO these are separate functions
// create gather_coeffs, gather_idx, and a templated impl function
std::vector<label> gather_to_owner(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator &comm,
    const CommCounts &comm_pattern,
    const scalar *send_buffer, scalar *recv_buffer);

std::vector<label> gather_to_owner(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator &comm,
    const CommCounts &comm_pattern,
    label size, const label *data, label offset = 0);

struct CommunicationPattern {
    using comm_size_type = label;

    gko::experimental::mpi::communicator comm;

    // an array storing to which rank to communicate
    gko::array<comm_size_type> target_ids;

    // an array storing how many elements to communicate
    // to the corresponding target_id
    gko::array<comm_size_type> target_sizes;

    // send_idx stores the index_set ie which cells
    // are owned by the interface and the corresponding target rank
    std::vector<std::pair<gko::array<label>, comm_size_type>> send_idxs;
};

std::ostream &operator<<(std::ostream &out, const CommunicationPattern &e);

// TODO move to Repartitioner
// TODO NOTE currently we have just one communication pattern for multiple proc
// interfaces
//  hence this will communicate vectors which are too long, FIXME potentially
//  have the wrong idxs and will communictate the same remote buffer several
//  times through the combination matrix.
CommunicationPattern repartition_comm_pattern(
    label ranks_per_gpu, CommunicationPattern &src_comm_pattern,
    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
        partition);
