// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include "OGL/CommunicationPattern.H"
#include "OGL/common.H"

namespace Foam {

/* @brief based on a comm_pattern consecutive ldu_mappings are computed
 *
 * After merging sparsity patterns during repartitioning the ldu mapping are
 * not consecutive ie they could look like [0 1 2 3 | 0 1 2 3 | 0 1], where
 * | is the former rank boundary. Based on the comm pattern we compute a new
 * mapping as  [ 0 1 2 3 | 4 5 6 7| 8 9 ] where the following offsets [ 0 |
 * 4 | 8 ] based on the recv_counts are used.
 *
 * */
void make_ldu_mapping_consecutive(const AllToAllPattern &comm_pattern,
                                  std::vector<label> &ldu_mapping, label rank,
                                  label ranks_per_gpu);

/* @brief computes the dimensions of square matrix based row index array
 *
 * @note assumes that rows are ordered
 * @params ordered array of row indices
 * returns dimensions
 * */
gko::dim<2> compute_dimensions(const std::vector<label> &rows);


/* The SparsityPattern holds row and column index data using gko::arrays.
 * This struct is used for easy generation of (distributed) ginkgo matrices.
 *
 * Additionally it keeps track which parts of the sparsity pattern belongs to
 * which interface.
 * */
struct SparsityPattern {
    /* constructor from vectors, assumes a single interface */
    SparsityPattern(std::shared_ptr<const gko::Executor> exec, gko::dim<2> dim_,
                    const std::vector<label> &rows,
                    const std::vector<label> &cols,
                    const std::vector<label> &mapping,
                    const std::vector<gko::span> &spans_,
                    const std::vector<label> &rank_)
        : num_nnz(rows.size()),
          row_idxs(exec, rows.begin(), rows.end()),
          col_idxs(exec, cols.begin(), cols.end()),
          ldu_mapping(exec, mapping.begin(), mapping.end()),
          dim(dim_),
          spans(spans_),
          rank(rank_)
    {
        // For every rank there should be a span
        ASSERT_EQ(spans.size(), rank.size());

        ASSERT_EQ(rows.size(), cols.size());
        ASSERT_EQ(rows.size(), mapping.size());
    }

    SparsityPattern(std::shared_ptr<const gko::Executor> exec)
        : num_nnz(0),
          row_idxs{exec},
          col_idxs{exec},
          ldu_mapping{exec},
          dim(gko::dim<2>{})
    {}

    SparsityPattern(std::shared_ptr<const gko::Executor> exec, const label size,
                    gko::dim<2> dim, std::vector<label> &rows,
                    std::vector<label> &cols, std::vector<label> &map,
                    std::vector<label> &spans_begin,
                    std::vector<label> &spans_end, std::vector<label> &ranks)
        : num_nnz(size),
          row_idxs{exec, static_cast<gko::size_type>(size)},
          col_idxs{exec, static_cast<gko::size_type>(size)},
          ldu_mapping{exec, static_cast<gko::size_type>(size)},
          dim(dim)
    {
        ASSERT_EQ(size, rows.size());
        std::copy(rows.begin(), rows.end(), row_idxs.get_data());

        ASSERT_EQ(size, cols.size());
        std::copy(cols.begin(), cols.end(), col_idxs.get_data());

        ASSERT_EQ(size, map.size());
        std::copy(map.begin(), map.end(), ldu_mapping.get_data());

        ASSERT_EQ(spans_begin.size(), spans_end.size());
        std::transform(spans_begin.begin(), spans_begin.end(),
                       spans_end.begin(), std::back_inserter(spans),
                       [](const auto begin, const auto end) {
                           return gko::span{static_cast<gko::size_type>(begin),
                                            static_cast<gko::size_type>(end)};
                       });

        ASSERT_EQ(spans_begin.size(), ranks.size());
        rank = ranks;
    }

    // TODO num_nnz makes no sense ->nnz;
    const label num_nnz;

    mutable gko::Array<label> row_idxs;

    mutable gko::Array<label> col_idxs;

    // ldu_mapping[sorted_(csr)_position] =
    // unsorted_(consecutive_ldu_)_position
    mutable gko::Array<label> ldu_mapping;

    // dimensions of the resulting matrix
    mutable gko::dim<2> dim;

    // A vector of spans indicating begin and end of each interface
    // this is used to keep all col and row idx consecutive in memory
    // TODO this can be refactored, since its original motivation
    // was to support the distributed_read function, which is not
    // needed anymore. However, removing would require row and col_idxs to
    // be stored in a std::vector<gko::Array>
    mutable std::vector<gko::span> spans;

    // keep track of original (local sparsity) or communication (non-local) rank
    // of each interface or matrix block
    mutable std::vector<label> rank;
};

}  // namespace Foam
