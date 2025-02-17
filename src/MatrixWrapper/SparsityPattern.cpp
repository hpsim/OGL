// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/SparsityPattern.hpp"

namespace detail {


std::pair<std::vector<std::vector<label>>, std::vector<label>> compress_cols(
    std::vector<std::vector<label>> in, std::vector<label> comm_id)
{
    // create a sorting map based on the comm ids
    // here the ids with higher id should receive data first
    auto id_permutation =
        sort_permutation(comm_id, [](label a, label b) { return a < b; });
    std::map<label, label> col_map;

    std::vector<label> global_cols;
    for (auto &col : in) {
        for (auto val : col) {
            global_cols.push_back(val);
        }
    }

    label ctr = 0;
    // iterate in the order of communication ranks
    for (auto id : id_permutation) {
        auto &cols = in[id];
        for (auto col : cols) {
            // new element found
            if (col_map.find(col) == col_map.end()) {
                // global_idx -> compressed
                col_map[col] = ctr;
            }
            ctr++;
        }
    }

    std::vector<label> map(ctr, 0);
    for (size_t i = 0; i < ctr; i++) {
        size_t index = col_map[global_cols[i]];
        map[index] = global_cols[i];
    }

    std::vector<std::vector<label>> ret;
    for (size_t i = 0; i < in.size(); i++) {
        std::vector<label> uncompressed(in[i]);
        std::vector<label> compressed;
        compressed.reserve(uncompressed.size());
        for (auto &val : uncompressed) {
            compressed.push_back(col_map[val]);
        }
        ret.push_back(compressed);
    }

    return {ret, map};
}


// std::pair<std::vector<std::vector<label>>, std::vector<label>> compress_cols(
//     std::vector<std::vector<label>> in, std::vector<label> orig_ids)
// {
//     std::map<label, label> col_map;
//     std::vector<label> global_cols;
//
//     for (auto &col : in) {
//         for (auto val : col) {
//             global_cols.push_back(val);
//         }
//     }
//
//     // it does not need to ordered by global cols, because it does not send
//     ordered
//     // it only sends ordered based on ranks
//     auto id_permutation =
//         sort_permutation(global_cols, [](label a, label b) { return a < b;
//         });
//
//     global_cols = apply_permutation(global_cols, id_permutation);
//
//     label ctr = 0;
//     for (auto col : global_cols) {
//         // new element found
//         if (col_map.find(col) == col_map.end()) {
//             col_map[col] = ctr;
//             // ctr++
//         }
//         // TODO
//         // NOTE the counter is increased in any case
//         // since we send all rows even if they are send multiple times
//         // we could send rows only once, by adapting Communication pattern
//         // but i guess this won't change much performance wise
//         ctr++;
//     }
//
//     std::vector<label> map(global_cols.size(), 0);
//     for (size_t i=0;i < global_cols.size();i++) {
//         size_t index = col_map[global_cols[i]];
//         map[index] = global_cols[i];
//     }
//
//
//     std::vector<std::vector<label>> ret;
//     for (auto &col : in) {
//         std::vector<label> compressed;
//         compressed.reserve(col.size());
//         for (auto val : col) {
//             compressed.push_back(col_map[val]);
//         }
//         ret.push_back(compressed);
//     }
//
//     return {ret, map};
// }

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
