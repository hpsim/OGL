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

std::ostream& operator<<(std::ostream& out, const CommunicationPattern& e) {
    out << "CommunicationPattern: for rank: " << e.comm.rank(); 
    out << " {"; 
    out << "\ntarget_ids: " << e.target_ids;
    out << "\ntarget_sizes: " << e.target_sizes;
    out << "}\n";
    return out;
}

CommunicationPattern repartition_comm_pattern(
    label ranks_per_gpu, 
    CommunicationPattern &src_comm_pattern)
{
    using comm_size_type = label;

    if (ranks_per_gpu == 1) {
        return CommunicationPattern(src_comm_pattern);
    }


    auto exec = src_comm_pattern.target_ids.get_executor();
    auto target_rank = [ranks_per_gpu](label src_rank, label dst_rank) {
        // all ranks 0 .. ranks_per_gpu go to 0
        return src_rank % ranks_per_gpu;
    };

    // find ranks which are now local
    std::vector<label> repart_target_ids{};
    std::vector<label> repart_target_sizes{};
    std::vector<std::pair<gko::array<label>, comm_size_type>> repart_send_idxs;
    label procs_ctr{0};
    label rank {src_comm_pattern.comm.rank()};
    for (int i = 0; i < src_comm_pattern.target_ids.get_size(); i++) {
        label target_id = src_comm_pattern.target_ids.get_data()[i];
        // check if communication partner is local after repartitioning
        // if it is not local we keep it
        if (target_rank(rank, target_id) != rank ) {
            repart_target_ids.push_back(target_id);
            repart_target_sizes.push_back(
                src_comm_pattern.target_sizes.get_data()[i]);
            repart_send_idxs.push_back(src_comm_pattern.send_idxs[i]);
            procs_ctr++;
        } else {
            // store reparted ranks
            std::cout <<  "!!! reparted rank [" << rank << "] " <<  src_comm_pattern.target_ids.get_data()[i] << "\n";

        }
        // we also have to get the communication partner from ranks that are now local
    }

    gko::array<comm_size_type> target_ids{exec, repart_target_ids.begin(),
                                          repart_target_ids.end()};
    gko::array<comm_size_type> target_sizes{exec, repart_target_sizes.begin(),
                                            repart_target_sizes.end()};

    auto ret =  CommunicationPattern{src_comm_pattern.comm, target_ids, target_sizes, repart_send_idxs};
    std::cout << ret;
    return ret;
}


