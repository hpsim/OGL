// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/SparsityPattern.H"

namespace Foam {

gko::dim<2> compute_dimensions(const gko::array<label> &rows)
{
    // helper function to get the last element of a ginkgo array
    auto get_back = [](const gko::array<label> &in) {
        return *(in.get_const_data() + in.get_size());
    };

    gko::size_type num_rows = get_back(rows) + 1;

    return gko::dim<2>{num_rows, num_rows};
}


void make_ldu_mapping_consecutive(const AllToAllPattern &comm_pattern,
                                  gko::array<label> &ldu_mapping, label rank,
                                  label ranks_per_gpu)
{
    // TODO check if ldu_mapping is on host exec
    label ldu_offset = 0;
    auto *data = ldu_mapping.get_data();

    for (label i = 0; i < ranks_per_gpu; i++) {
        auto size = comm_pattern.recv_counts[i];
        std::transform(data + ldu_offset, data + ldu_offset + size,
                       data + ldu_offset,
                       [&](label idx) { return idx + ldu_offset; });
        ldu_offset += size;
    }
}

}  // namespace Foam
