// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"

#include "CommunicationPattern.H"
#include "MatrixWrapper/SparsityPattern.H"

namespace detail {

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
    const std::vector<label> &idx, const std::vector<gko::span> &spans,
    const std::vector<label> &ranks)
{
    std::vector<label> ret;
    ret.reserve(idx.size());

    for (label i = 0; i < ranks.size(); i++) {
        auto rank = ranks[i];
        auto [begin, end] = spans[i];
        label offset = partition->get_range_bounds()[rank];
        for (label j = begin; j < end; j++) {
            ret.push_back(idx.data()[j] + offset);
        }
    }
    return ret;
}

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
    std::vector<label> &in, label rank)
{
    label offset = partition->get_range_bounds()[rank];
    std::transform(in.begin(), in.end(), in.begin(),
                   [&](label idx) { return idx - offset; });
}

/* @brief exchange spans and rank between owner and non-owner ranks
**
** returns a pair vector spans
*/
std::pair<std::vector<gko::span>, std::vector<label>> exchange_span_ranks(
    const ExecutorHandler &exec_handler, label ranks_per_gpu,
    const std::vector<gko::span> &spans, const std::vector<label> &ranks);

template <typename T>
std::vector<T> apply_permutation(const std::vector<T> vec,
                                 const std::vector<label> &p);

/* Sorts a vector and returns the permutation for reuse
 */
template <typename T, typename Compare>
std::vector<label> sort_permutation(const std::vector<T> &vec, Compare compare);

}  // namespace detail

/** @class Collects functions to repartition communication patterns
 * and matrices. Here repartitioning refers to changing a given partitioning on
 * n ranks to m ranks with n>=m
 */
class Repartitioner {
private:
    using partition = gko::experimental::distributed::Partition<label, label>;

    const label size_;  //! Size (n matrix rows) before repartitioning

    const label repart_size_;  //! Size after repartitioning

    const label ranks_per_gpu_;

    const label verbose_;

    std::shared_ptr<const partition> orig_partition_;

public:
    Repartitioner(label size, label ranks_per_gpu, label verbose,
                  const ExecutorHandler &exec_handler)
        : size_(size),
          repart_size_(Repartitioner::compute_repart_size(size, ranks_per_gpu,
                                                          exec_handler)),
          ranks_per_gpu_(ranks_per_gpu),
          verbose_(verbose),
          orig_partition_(gko::share(
              gko::experimental::distributed::build_partition_from_local_size<
                  label, label>(exec_handler.get_ref_exec(),
                                *exec_handler.get_communicator().get(),
                                size))){};

    /* returns the owner rank for a given rank */
    label get_owner_rank(label rank) const
    {
        return compute_owner_rank(rank, ranks_per_gpu_);
    };

    /* returns the owner rank for a given rank */
    label get_owner_rank(const ExecutorHandler &exec_handler) const
    {
        return get_owner_rank(exec_handler.get_rank());
    };

    /* returns if current rank is an owner  */
    bool is_owner(const ExecutorHandler &exec_handler) const
    {
        return exec_handler.get_rank() ==
               get_owner_rank(exec_handler.get_rank());
    };

    /* @brief check if the given rank gets local after repartitioning
     *
     * */
    bool reparts_to_local(const ExecutorHandler &exec_handler, label rank) const
    {
        return get_owner_rank(exec_handler) ==
               compute_owner_rank(rank, ranks_per_gpu_);
    };

    label get_ranks_per_gpu() const { return ranks_per_gpu_; }

    /* @brief computes the size of the submatrix owned by a given rank after
     * repartitioning
     *
     * Given a local size (can represent nrows or nnzs) the new size of
     * this rank after repartitioning gets computed
     */
    static label compute_repart_size(label local_size, label ranks_per_gpu,
                                     const ExecutorHandler &exec_handler);

    label get_repart_size() const { return repart_size_; }

    // TODO pass original dim as argument, if size can also be nnz
    gko::dim<2> get_repart_dim() const
    {
        return gko::dim<2>{static_cast<gko::size_type>(repart_size_),
                           static_cast<gko::size_type>(repart_size_)};
    }

    std::shared_ptr<const partition> get_orig_partition() const
    {
        return orig_partition_;
    }


    /* @brief given received interfaces this function sorts interfaces into
     *
     *local and non local returns: SparsityPattern storing new non_local
     *sparsity pattern, and a pair of locality information, where the first bool
     *stores if the interface performs a local communication and the second bool
     *if the the interface was originally from this rank
     *
     */
    std::vector<bool> build_non_local_interfaces(
        const ExecutorHandler &exec_handler,
        std::shared_ptr<
            const gko::experimental::distributed::Partition<label, label>>
            partition,
        std::vector<label> &local_rows, std::vector<label> &local_cols,
        std::vector<label> &local_mapping, std::vector<label> &local_ranks,
        std::vector<gko::span> &local_spans, std::vector<label> &non_local_rows,
        std::vector<label> &non_local_cols,
        std::vector<label> &non_local_mapping,
        std::vector<label> &non_local_ranks,
        std::vector<label> &non_local_rank_origin,
        std::vector<gko::span> &non_local_spans) const;

    /* @brief function to repartition a sparsity pattern
     *
     * function to repartition a sparsity pattern based on ranks_per_gpu
     * ie changing a sparsity pattern which has rows on each rank to a sparsity
     * pattern which has rows only on every ranks_per_gpu-th rank (owner)
     *
     * @param exec_handler the executor handler
     * @param src_local_pattern the original sparsity pattern of the local
     *matrix
     * @param src_non_local_pattern the original sparsity pattern of the non
     *local matrix
     * returns the new sparsity patterns (local and non-local) and a vector
     * tracking the interfaces. This contains a pair where the first entry
     *(bool)
     * signals whether this is a new local interface (no communication), and
     *the
     * second entry (label) stores the original rank of the interface
     */
    std::tuple<std::shared_ptr<SparsityPattern>,
               std::shared_ptr<SparsityPattern>,
               std::vector<std::pair<bool, label>>>
    repartition_sparsity(
        const ExecutorHandler &exec_handler,
        std::shared_ptr<const SparsityPattern> src_local_pattern,
        std::shared_ptr<const SparsityPattern> src_non_local_pattern) const;

    /* @brief repartition a communication pattern
     * function to repartition a communication pattern based on ranks_per_gpu
     * ie changing a communication pattern which has rows on each rank and where
     * every rank communicates to a communication pattern which has rows only on
     * every ranks_per_gpu-th rank and thus only communicates between owner
     * ranks
     *
     * @param src_comm_pattern the original communication pattern of the current
     * rank belonging to the distributed matrix
     * @param partition the original partition belonging to the distributed
     * matrix returns the repartitioned communcation pattern
     */
    std::shared_ptr<const CommunicationPattern> repartition_comm_pattern(
        const ExecutorHandler &exec_handler,
        std::shared_ptr<const CommunicationPattern> src_comm_pattern,
        std::shared_ptr<
            const gko::experimental::distributed::Partition<label, label>>
            partition) const;
};
