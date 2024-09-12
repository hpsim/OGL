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
    const label *idx, const std::vector<gko::span> &spans,
    const std::vector<label> &ranks);

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

/* @brief exchange spans and rank between owner and non-owner ranks
**
** returns a pair vector spans
*/
std::tuple<std::vector<gko::span>, std::vector<label>, std::vector<label>>
exchange_spans_ranks(const ExecutorHandler &exec_handler, label ranks_per_gpu,
                     const std::vector<gko::span> &spans,
                     const std::vector<label> &src_ranks);

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

    const label size_;  //! Local size (matrix rows) before repartitioning

    const label repart_size_;  //! Local size (matrix rows) after repartitioning

    const label ranks_per_gpu_;

    const bool fuse_;  //! Whether to merge all interfaces into a single
                       //! local/non_local mtx

    const label verbose_;

    std::shared_ptr<const partition> orig_partition_;

public:
    Repartitioner(label size, label ranks_per_gpu, label verbose,
                  const ExecutorHandler &exec_handler, bool fuse)
        : size_(size),
          repart_size_(Repartitioner::compute_repart_size(size, ranks_per_gpu,
                                                          exec_handler)),
          ranks_per_gpu_(ranks_per_gpu),
          fuse_(fuse),
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

    bool get_fused() const { return fuse_; }

    /* @brief computes the size of the submatrix owned by a given rank after
     * repartitioning
     *
     * Given a local size (can represent nrows or nnzs) the new size of
     * this rank after repartitioning gets computed
     */
    static label compute_repart_size(label local_size, label ranks_per_gpu,
                                     const ExecutorHandler &exec_handler);

    label get_orig_size() const { return size_; }

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

    /* @brief given gathered sparsity information copy local interfaces
     *
     * After gathering local and non-local rows, cols, and mapping
     * we have interfaces that are now local in the non-local rows, cols and
     * mapping. Thus this function copies the rows, cols and mapping, spans etc
     * to the corresponding local vectors.
     * returns a vector of locality information.
     * @param non_local_cols vector in global local indices
     */
    std::vector<std::tuple<bool, label, label>> build_non_local_interfaces(
        const ExecutorHandler &exec_handler,
        std::shared_ptr<
            const gko::experimental::distributed::Partition<label, label>>
            partition,
        std::vector<label> &local_rows, std::vector<label> &local_cols,
        std::vector<label> &local_mapping, std::vector<gko::span> &local_spans,
        std::vector<label> &non_local_rows, std::vector<label> &non_local_cols,
        std::vector<label> &non_local_mapping,
        std::vector<label> &non_local_rank_origin,
        std::vector<gko::span> &non_local_spans,
        std::vector<label> &src_non_local_target_ids) const;

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
     * second entry (label) tracks the original rank of the interface
     * @param fuse whether to combine all submatrices
     */
    std::tuple<std::shared_ptr<SparsityPattern>,
               std::shared_ptr<SparsityPattern>,
               std::vector<std::tuple<bool, label, label>>>
    repartition_sparsity(
        const ExecutorHandler &exec_handler,
        std::shared_ptr<const SparsityPattern> src_local_pattern,
        std::shared_ptr<const SparsityPattern> src_non_local_pattern,
        std::vector<label> src_non_local_target_ids, bool fuse) const;

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
     * matrix returns the repartitioned communication pattern
     */
    std::shared_ptr<const CommunicationPattern> repartition_comm_pattern(
        const ExecutorHandler &exec_handler,
        std::shared_ptr<const CommunicationPattern> src_comm_pattern) const;
};
