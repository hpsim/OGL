// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/SparsityPattern.H"

namespace Foam {

gko::dim<2> compute_dimensions(const std::vector<label> &rows)
{
    gko::size_type num_rows = rows.back() + 1;
    return gko::dim<2>{num_rows, num_rows};
}

void compress_cols(gko::array<label> &in)
{
    for (size_t i = 0; i < in.get_size(); i++) {
        in.get_data()[i] = i;
    }
}

void make_ldu_mapping_consecutive(const AllToAllPattern &comm_pattern,
                                  std::vector<label> &ldu_mapping, label rank,
                                  label ranks_per_gpu)
{
    label ldu_offset = 0;
    auto *data = ldu_mapping.data();

    sleep(rank);
    std::cout << __FILE__ << " rank " << rank << "mapping before" << ldu_mapping << "\n";

    for (label i = 0; i < ranks_per_gpu; i++) {
        auto size = comm_pattern.recv_counts[rank + i];
        std::transform(data + ldu_offset, data + ldu_offset + size,
                       data + ldu_offset,
                       [&](label idx) { return idx + ldu_offset; });
        ldu_offset += size;
    }
    std::cout << __FILE__ << " rank " << rank << "mapping after" << ldu_mapping << "\n";
}

}  // namespace Foam
