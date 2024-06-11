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


Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    HostMatrix.H

\*---------------------------------------------------------------------------*/
#pragma once

/* Vector variant of the SparsityPattern
**
*/
struct SparsityPatternVector {
    std::vector<label> rows;
    std::vector<label> cols;
    std::vector<label> mapping;
    std::vector<label> begin;
    std::vector<label> end;
    std::vector<label> ranks;
};

struct SparsityPattern {
    SparsityPattern(std::shared_ptr<const gko::Executor> exec)
        : size_(0),
          row_idxs{exec, static_cast<gko::size_type>(size_)},
          col_idxs{exec, static_cast<gko::size_type>(size_)},
          ldu_mapping{exec, static_cast<gko::size_type>(size_)},
          dim(gko::dim<2>{0, 0}),
          interface_spans(std::vector<gko::span>{}),
          rank(std::vector<label>{})
    {}

    SparsityPattern(std::shared_ptr<const SparsityPattern> other)
        : size_(other->size_),
          row_idxs(other->row_idxs),
          col_idxs(other->col_idxs),
          ldu_mapping(other->ldu_mapping),
          dim(other->dim),
          interface_spans(other->interface_spans),
          rank(other->rank)
    {}

    SparsityPattern(std::shared_ptr<const gko::Executor> exec, label size)
        : size_(size),
          row_idxs{exec, static_cast<gko::size_type>(size_)},
          col_idxs{exec, static_cast<gko::size_type>(size_)},
          ldu_mapping{exec, static_cast<gko::size_type>(size_)}
    {}

    SparsityPattern(std::shared_ptr<const gko::Executor> exec, gko::dim<2> dim,
                    const SparsityPatternVector &in)
        : SparsityPattern(exec, in.rows.size(), dim, in.rows, in.cols,
                          in.mapping, in.begin, in.end, in.ranks) {}

    SparsityPattern(std::shared_ptr<const gko::Executor> exec, const label size,
                    gko::dim<2> dim, const std::vector<label> &rows,
                    const std::vector<label> &cols,
                    const std::vector<label> &map,
                    const std::vector<label> &spans_begin,
                    const std::vector<label> &spans_end,
                    const std::vector<label> &ranks)
        : size_(size),
          dim(dim),
          row_idxs{exec, static_cast<gko::size_type>(size)},
          col_idxs{exec, static_cast<gko::size_type>(size)},
          ldu_mapping{exec, static_cast<gko::size_type>(size)}
    {
        ASSERT_EQ(size, rows.size());
        std::copy(rows.begin(), rows.end(), row_idxs.get_data());

        ASSERT_EQ(size, cols.size());
        std::copy(cols.begin(), cols.end(), col_idxs.get_data());

        ASSERT_EQ(size, map.size());
        std::copy(map.begin(), map.end(), ldu_mapping.get_data());

        ASSERT_EQ(spans_begin.size(), spans_end.size());
        std::transform(spans_begin.begin(), spans_begin.end(),
                       spans_end.begin(), std::back_inserter(interface_spans),
                       [](const auto begin, const auto end) {
                           return gko::span{static_cast<gko::size_type>(begin),
                                            static_cast<gko::size_type>(end)};
                       });

        ASSERT_EQ(spans_begin.size(), ranks.size());
        rank = ranks;
    }

    // TODO rename the public members and remove trailing _
    // number of nnz entries
    const label size_;

    mutable gko::Array<label> row_idxs;

    mutable gko::Array<label> col_idxs;

    // ldu_mapping[sorted_(csr)_position] =
    // unsorted_(consecutive_ldu_)_position
    mutable gko::Array<label> ldu_mapping;

    // dimensions of the resulting matrix
    mutable gko::dim<2> dim;

    // A vector of spans indicating begin and end of each interface
    // this is used to keep all col and row idx consecutive in memory
    mutable std::vector<gko::span> interface_spans;

    // keep track of communication rank id of each interface or matrix
    // block
    mutable std::vector<label> rank;
};

// struct SparsityDataMap {

//     SparsityDataMap (
//         const SparsityPattern& pattern_in,
//     {}

//     const SparsityPattern& pattern;

//     const vector<scalar*> data;

// };
