// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include "OGL/CommunicationPattern.hpp"
#include "OGL/common.hpp"


namespace detail {

/* @brief applies permutation vector s = v[p[i]]
 *                 e.g. s[0] = v[p[0] -> 4 ]
 */
template <typename T>
std::vector<T> apply_permutation(const std::vector<T> vec,
                                 const std::vector<label> &p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
                   [&](label i) { return vec[i]; });
    return sorted_vec;
}

template <typename T, typename Compare>
std::vector<label> sort_permutation(const std::vector<T> &vec, Compare compare)
{
    std::vector<label> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::stable_sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j) {
        return compare(vec[i], vec[j]);
    });
    return p;
}

/* @brief compute compressed column indices
**
** based on the global column indices this function computes the compressed
*column indices
** here compressed columns are column indices starting at 0 and corresponding to
*the order how non-local vector data is received.
**
**@param in - vector of vectors of global column indices per
*interface/sub-matrix
**@param comm_id - vector of with the ranks to which each interface communicates
*to
*/
std::pair<std::vector<std::vector<label>>, std::vector<label>> compress_cols(
    std::vector<std::vector<label>> in, std::vector<label> comm_rank);
}  // namespace detail

namespace Foam {

/* Convert to global
** Given an array of column indices local to the communication rank
** this function offsets these array
** @param idx pointer to gko::array holding the indices which need to be
*converted from local to global ids
** @param spans start and ends of the interfaces
** @param ranks the rank to which the interface index is a local row index
*/
std::vector<label> convert_to_global(
    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
        partition,
    const label *idx, const label size, const label rank);

/* Convert to local
** Given an array of column indices local to the communication rank
** this function offsets these array
** @param idx pointer to gko::array holding the indices which need to be
*converted from local to global ids
** @param spans start and ends of the interfaces
** @param ranks the rank to which the interface index is a local row index
*/
void convert_to_local(
    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
        partition,
    std::vector<label> &in, label rank);


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
class SparsityPattern {
public:
    SparsityPattern()
        : nnz_(), rows_(), cols_(), map_(), id_(), orig_rank_(), comm_rank_()
    {}


    /* @brief given rows and cols a new interface/submatrix is added and stored
     *  in row major format,
     *
     *  @row_major_order - whether the passed interface is already in
     * row_major_order
     *  */
    void insert_interface(std::vector<label> &&rows, std::vector<label> &&cols,
                          label orig_rank, label comm_rank, label id,
                          label comm_id, bool row_major_order = true)
    {
        OGL_ASSERT_EQ(rows.size(), cols.size());

        std::vector<label> map(rows.size());
        std::iota(map.begin(), map.end(), 0);
        if (!row_major_order) {
            sort_sparsity(rows, cols, map);
        }

        nnz_ += rows.size();
        rows_.push_back(rows);
        cols_.push_back(cols);

        id_.push_back(id);
        comm_id_.push_back(comm_id);
        orig_rank_.push_back(orig_rank);
        comm_rank_.push_back(comm_rank);
        orig_comm_rank_.push_back(comm_rank);

        map_.push_back(map);
    }

    void insert_interface(std::vector<label> &&rows, std::vector<label> &&cols,
                          std::vector<label> &&map, label id, label comm_id,
                          label orig_rank, label comm_rank)
    {
        OGL_ASSERT_EQ(rows.size(), cols.size());

        nnz_ += rows.size();
        rows_.push_back(rows);
        cols_.push_back(cols);
        map_.push_back(map);
        id_.push_back(id);
        comm_id_.push_back(comm_id);
        orig_rank_.push_back(orig_rank);
        comm_rank_.push_back(comm_rank);
        orig_comm_rank_.push_back(comm_rank);
    }

    std::vector<label> compute_to_global_map(bool fuse) const
    {
        // TODO just change the sorting algorithm of compress_cols
        std::vector<label> inv_comm_ranks;
        for (auto rank : comm_rank_) {
            inv_comm_ranks.push_back(-rank);
        }

        return std::get<1>(detail::compress_cols(cols_, orig_comm_rank_));
    }

    // TODO could make this a free function
    /* @brief returns rows, columns and mapping and id for consumption on create
     * distributed
     *
     * @returns a tuple of local_rows, local_cols, local_map, local_ids
     */
    std::tuple<std::vector<std::vector<label>>, std::vector<std::vector<label>>,
               std::vector<std::vector<label>>, std::vector<label>>
    get_vecs(bool compress_cols, bool repartioned)
    {
        // early return
        // if not repartitioned no ldu interfaces to fuse
        if (!repartioned) {
            return {rows_,
                    (compress_cols)
                        ? std::get<0>(detail::compress_cols(cols_, comm_id_))
                        : cols_,
                    map_, id_};
        }

        label size_upper;  // nnz on interface 0
        label size_diag;   // nnz on interface 2

        for (size_t i = 0; i < id_.size(); i++) {
            auto id = id_[i];
            // count only ldu interfaces
            if (id < 0) continue;

            if (id % 3 == 0) {
                size_upper += rows_[i].size();
            }
            if (id_[i] % 3 == 2) {
                size_diag += rows_[i].size();
            }
        }

        auto fuse_ldu = [size_upper, size_diag](auto &vec, auto &id,
                                                bool offset) {
            /* given a vec and a predicate function this function pushes to out
             * if pred is true */
            auto transform_if = [offset](auto &vec, auto pred, auto &out) {
                size_t ctr{0};
                // vec is vec<vec>
                for (size_t i = 0; i < vec.size(); i++) {
                    if (pred(i)) {
                        auto &v = vec[i];
                        for (size_t j = 0; j < v.size(); j++) {
                            out.push_back(v[j] + ctr);
                        }
                        ctr += (offset) ? v.size() : 0;
                    }
                }
            };

            std::vector<label> upper;
            // upper.reserve(size_upper);
            std::vector<label> lower;
            // lower.reserve(size_upper);
            std::vector<label> diag;
            // diag.reserve(size_diag);

            transform_if(
                vec, [id](size_t i) { return id[i] >= 0 && id[i] % 3 == 0; },
                upper);
            transform_if(
                vec, [id](size_t i) { return id[i] >= 0 && id[i] % 3 == 1; },
                lower);
            transform_if(
                vec, [id](size_t i) { return id[i] >= 0 && id[i] % 3 == 2; },
                diag);

            std::vector<std::vector<label>> ret{upper, lower, diag};

            for (size_t i = 0; i < vec.size(); i++) {
                if (id[i] < 0) {
                    ret.push_back(vec[i]);
                }
            }

            return ret;
        };

        std::vector<label> ret_id{0, 1, 2};

        for (auto id : id_) {
            if (id < 0) {
                ret_id.push_back(id);
            }
        }

        return {fuse_ldu(rows_, id_, false), fuse_ldu(cols_, id_, false),
                fuse_ldu(map_, id_, true), ret_id};
    }

    // TODO could make this a free function
    /* @brief returns rows, columns and mapping and id for consumption on create
     * distributed
     *
     * @returns a tuple of local_rows, local_cols, local_map, local_ids
     */
    std::tuple<std::vector<std::vector<label>>, std::vector<std::vector<label>>,
               std::vector<std::vector<label>>, std::vector<label>>
    get_fused_vecs(bool compress_cols)
    {
        size_t reserve_size{0};
        for (size_t i = 0; i < id_.size(); i++) {
            reserve_size += rows_[i].size();
        }

        std::vector<label> rows;
        rows.reserve(reserve_size);
        std::vector<label> out_cols;
        out_cols.reserve(reserve_size);
        std::vector<label> map;
        map.reserve(reserve_size);

        label map_offset = 0;

        auto cols =
            (compress_cols)
                ? std::get<0>(detail::compress_cols(cols_, orig_comm_rank_))
                : cols_;

        // ldu part first
        if (!compress_cols) {
            for (int id = 0; id < 3; id++) {
                for (size_t i = 0; i < id_.size(); i++) {
                    if (id_[i] != id) {
                        continue;
                    }
                    rows.insert(rows.end(), rows_[i].begin(), rows_[i].end());
                    out_cols.insert(out_cols.end(), cols[i].begin(),
                                    cols[i].end());
                    size_t iface_length = rows_[i].size();
                    for (size_t j = 0; j < iface_length; j++) {
                        map.push_back(map_[i][j] + map_offset);
                    }
                    map_offset += iface_length;
                }
            }
        }

        // the rest. here the ids are negative
        for (size_t i = 0; i < id_.size(); i++) {
            if (id_[i] >= 0) {
                continue;
            }
            rows.insert(rows.end(), rows_[i].begin(), rows_[i].end());
            out_cols.insert(out_cols.end(), cols[i].begin(), cols[i].end());
            size_t iface_length = rows_[i].size();
            for (size_t j = 0; j < iface_length; j++) {
                map.push_back(map_[i][j] + map_offset);
            }
            map_offset += iface_length;
        }

        sort_sparsity(rows, out_cols, map);

        return {{rows}, {out_cols}, {map}, {(compress_cols) ? -1 : 0}};
    }

    /* @brief move all interfaces to this data structure that communicate to
     * a given rank
     *
     * @param comm_rank move if other.comm_rank == comm_rank
     *  */
    template <typename Func>
    void move_interface(std::shared_ptr<SparsityPattern> other, label comm_rank,
                        Func convert_cols)
    {
        auto &other_rows = other->get_rows();
        auto &other_map = other->get_map();
        auto &other_cols = other->get_cols();
        auto &other_id = other->get_id();
        auto &other_comm_id = other->get_comm_id();
        auto &other_orig_comm_rank = other->get_orig_comm_rank();
        auto &other_orig_rank = other->get_orig_rank();
        auto &other_comm_rank = other->get_comm_rank();
        auto &other_nnz = other->get_nnz();

        // move to own ie local sparsity
        std::vector<label> del_from_other{};
        for (int i = 0; i < other_rows.size(); i++) {
            if (other_comm_rank[i] == comm_rank) {
                nnz_ += other_rows[i].size();
                rows_.push_back(std::move(other_rows[i]));
                cols_.push_back(convert_cols(std::move(other_cols[i])));
                map_.push_back(std::move(other_map[i]));
                comm_rank_.push_back(std::move(other_comm_rank[i]));
                orig_rank_.push_back(std::move(other_orig_rank[i]));
                id_.push_back(std::move(other_id[i]));
                comm_id_.push_back(std::move(other_comm_id[i]));
                orig_comm_rank_.push_back(std::move(other_orig_comm_rank[i]));
                del_from_other.push_back(i);
            }
        }

        // delete from other
        label num_del = del_from_other.size();
        for (int i = 0; i < num_del; i++) {
            label del = del_from_other[num_del - i - 1];
            // other_nnz -= other_rows[del].size();
            other_rows.erase(other_rows.begin() + del);
            other_cols.erase(other_cols.begin() + del);
            other_map.erase(other_map.begin() + del);
            other_comm_rank.erase(other_comm_rank.begin() + del);
            other_orig_comm_rank.erase(other_orig_comm_rank.begin() + del);
            other_orig_rank.erase(other_orig_rank.begin() + del);
            other_id.erase(other_id.begin() + del);
            other_comm_id.erase(other_comm_id.begin() + del);
        }

        other->get_nnz() = 0;
        for (auto &row : other_rows) {
            other->get_nnz() += row.size();
        }
    }

    // getter
    std::vector<std::vector<label>> &get_rows() { return rows_; }

    const std::vector<std::vector<label>> &get_rows() const { return rows_; }

    std::vector<std::vector<label>> &get_map() { return map_; }

    const std::vector<std::vector<label>> &get_map() const { return map_; }

    std::vector<std::vector<label>> &get_cols() { return cols_; }

    const std::vector<std::vector<label>> &get_cols() const { return cols_; }

    std::vector<label> &get_id() { return id_; }

    const std::vector<label> &get_id() const { return id_; }

    std::vector<int> get_lengths()
    {
        std::vector<label> ret{};
        for (auto &row : rows_) {
            ret.push_back(row.size());
        }
        return ret;
    }

    std::vector<label> &get_orig_rank() { return orig_rank_; }

    std::vector<label> &get_comm_rank() { return comm_rank_; }

    std::vector<label> &get_orig_comm_rank() { return orig_comm_rank_; }

    std::vector<label> &get_comm_id() { return comm_id_; }

    label &get_nnz() { return nnz_; }

    label get_nnz() const { return nnz_; }


private:
    label nnz_;

    std::vector<std::vector<label>> rows_;

    std::vector<std::vector<label>> cols_;

    // ldu_mapping is used to reorder copied matrix coefficients
    // to match the row major ordering of the sparsity pattern
    // in both cases (local and non local sparsity) the full list of
    // interfaces is reordered at once. thus all indices from [0, end)
    // should be present. ldu_mapping[sorted_(csr)_position] =
    // unsorted_(consecutive_ldu_)_position
    std::vector<std::vector<label>> map_;

    // map to HostMatrix.get_interface(id)
    std::vector<label> id_;

    // the id of the interface with which this interface communicates
    std::vector<label> comm_id_;

    // the rank from which this interface originates
    std::vector<label> orig_rank_;

    // the current rank with which this rank communicates
    std::vector<label> comm_rank_;

    // the current rank with which this rank communicated
    // originally
    std::vector<label> orig_comm_rank_;

    /* @brief sort given rows, cols, and map to be in row major order
     *
     * @row_major whether to sort it in row major (true) or column major order
     */
    void sort_sparsity(std::vector<label> &rows, std::vector<label> &cols,
                       std::vector<label> &map, bool row_major = true)
    {
        // add offset to mapping
        // so interface mapping is not continuous
        std::vector<label> permutation(rows.size());
        std::iota(permutation.begin(), permutation.end(), 0);
        if (row_major) {
            std::stable_sort(permutation.begin(), permutation.end(),
                             [&](std::size_t i, std::size_t j) {
                                 return std::tie(rows[i], cols[i]) <
                                        std::tie(rows[j], cols[j]);
                             });
        } else {
            std::stable_sort(permutation.begin(), permutation.end(),
                             [&](std::size_t i, std::size_t j) {
                                 return std::tie(cols[i], rows[i]) <
                                        std::tie(cols[j], rows[j]);
                             });
        }
        rows = detail::apply_permutation(rows, permutation);
        cols = detail::apply_permutation(cols, permutation);
        map = detail::apply_permutation(map, permutation);
    }
};

}  // namespace Foam
