// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"

#include "CommunicationPattern.H"
#include "MatrixWrapper/SparsityPattern.H"

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
                                size))) {};

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

    /* @brief computes the size of the submatrix owned by a given rank after repartitioning
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
    };


    /* @brief given received interfaces this function sorts into local and non
     *local
     * returns: SparsityPattern storing new non_local sparsity pattern,
     *  and a pair of locality information, where the first bool stores if the
     *  interface performs a local communication and the second bool if the
     *  the interface was originally from this rank
     */
    std::pair<SparsityPattern, std::vector<bool>> build_non_local_interfaces(
        const ExecutorHandler &exec_handler, SparsityPattern &loc,
        const SparsityPattern &non_loc) const;


    /*
    ** returns the new sparsity patterns (local and non-local) and a vector
    ** tracking the interfaces. This contains a pair where the first entry
    *(bool)
    ** signals whether this is a new local interface (no communication), and the
    ** second entry (label) tracks the original rank of the interface
    */
    std::tuple<std::shared_ptr<SparsityPattern>,
               std::shared_ptr<SparsityPattern>,
               std::vector<std::pair<bool, label>>>
    repartition_sparsity(
        const ExecutorHandler &exec_handler,
        std::shared_ptr<SparsityPattern> src_local_pattern,
        std::shared_ptr<SparsityPattern> src_non_local_pattern) const;

    /* @brief repartition a communication pattern
    **
    ** returns the repartitioned communcation pattern
    */
    std::shared_ptr<const CommunicationPattern> repartition_comm_pattern(
        const ExecutorHandler &exec_handler,
        std::shared_ptr<const CommunicationPattern> src_comm_pattern,
        std::shared_ptr<
            const gko::experimental::distributed::Partition<label, label>>
            partition) const;
};
