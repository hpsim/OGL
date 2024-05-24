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

struct SparsityPattern {
    SparsityPattern(std::shared_ptr<const gko::Executor> exec, const label size)
        : size_(size),
          row_idxs{exec, static_cast<gko::size_type>(size)},
          col_idxs{exec, static_cast<gko::size_type>(size)},
          ldu_mapping{exec, static_cast<gko::size_type>(size)}
    {}

    SparsityPattern(std::shared_ptr<const gko::Executor> exec, const label size,
                    gko::dim<2> dim, std::vector<label> &rows,
                    std::vector<label> &cols, std::vector<label> &map,
                    std::vector<label> &spans_begin,
                    std::vector<label> &spans_end, std::vector<label> &ranks)
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

    // keep track of original (communication) rank of each interface or matrix
    // block
    mutable std::vector<label> rank;
};
