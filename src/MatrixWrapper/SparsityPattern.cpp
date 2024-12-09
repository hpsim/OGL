// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/SparsityPattern.hpp"

namespace detail {

std::pair<std::vector<std::vector<label>>, std::vector<label>> compress_cols(
    std::vector<std::vector<label>> in, std::vector<label> ids)
{
    auto id_permutation =
        sort_permutation(ids, [](label a, label b) { return a < b; });
    std::map<label, label> col_map;


    label ctr = 0;
    for (auto id : id_permutation) {
        auto &cols = in[id];
        for (auto col : cols) {
            // new element found
            if (col_map.find(col) == col_map.end()) {
                // global_idx -> compressed
                col_map[col] = ctr;
                ctr++;
            }
        }
    }

    std::vector<label> map(col_map.size(), 0);
    for (auto [key, value] : col_map) {
        map[value] = key;
    }

    std::vector<std::vector<label>> ret;
    for (auto id : id_permutation) {
        // TODO std::transform would be better
        std::vector<label> uncompressed(in[id]);
        std::vector<label> compressed;
        compressed.reserve(uncompressed.size());
        for (auto &val : uncompressed) {
            compressed.push_back(col_map[val]);
        }
        ret.push_back(compressed);
    }


    return {ret, map};
}

}  // namespace detail

namespace Foam {

gko::dim<2> compute_dimensions(const std::vector<label> &rows)
{
    gko::size_type num_rows = rows.back() + 1;
    return gko::dim<2>{num_rows, num_rows};
}


std::vector<label> convert_to_global(
    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
        partition,
    const label *idx, const label size, const label rank)
{
    std::vector<label> ret;
    ret.reserve(size);
    label offset = partition->get_range_bounds()[rank];
    for (size_t i = 0; i < size; i++) {
        ret.push_back(idx[i] + offset);
    }
    return ret;
}


void convert_to_local(
    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
        partition,
    std::vector<label> &in, label rank)

{
    label offset = partition->get_range_bounds()[rank];
    std::transform(in.begin(), in.end(), in.begin(),
                   [&](label idx) { return idx - offset; });
}


void make_ldu_mapping_consecutive(const AllToAllPattern &comm_pattern,
                                  std::vector<label> &ldu_mapping, label rank,
                                  label ranks_per_gpu)
{
    label ldu_offset = 0;
    auto *data = ldu_mapping.data();

    for (label i = 0; i < ranks_per_gpu; i++) {
        auto size = comm_pattern.recv_counts[rank + i];
        std::transform(data + ldu_offset, data + ldu_offset + size,
                       data + ldu_offset,
                       [&](label idx) { return idx + ldu_offset; });
        ldu_offset += size;
    }
}

}  // namespace Foam
