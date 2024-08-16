// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/SparsityPattern.H"

void make_ldu_mapping_consecutive(const AllToAllPattern &comm_pattern,
                                  gko::array<label> &ldu_mapping, label rank,
                                  label ranks_per_gpu)
{
    label ldu_offset = 0;
    auto *data = ldu_mapping.get_data();
    // TODO check if ldu_mapping is on host exec

    for (label i = 0; i < ranks_per_gpu; i++) {
        auto size = comm_pattern.recv_counts[i];
        std::transform(data + ldu_offset, data + ldu_offset + size,
                       data + ldu_offset,
                       [&](label idx) { return idx + ldu_offset; });
        ldu_offset += size;
    }
}
