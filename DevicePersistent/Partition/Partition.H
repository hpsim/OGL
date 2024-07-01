// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>

#include <ginkgo/ginkgo.hpp>

#include "src/CommunicationPattern/CommunicationPattern.H"
#include "DevicePersistent/Base/Base.H"
#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"
#include "common/common.H"

namespace Foam {

struct PartitionInitFunctor {
    using comm_size_type = int;
    using part_type = gko::experimental::distributed::Partition<label, label>;

    const ExecutorHandler &exec_;

    const label local_size_;

    const label verbose_;

    const int ranks_per_gpu_;

    // Array storing the ranks of the neighbouring processors
    const gko::array<comm_size_type> target_ids_;

    // Array storing the ranks of the neighbouring processors
    const gko::array<comm_size_type> target_sizes_;

    const std::vector<std::pair<gko::array<label>, comm_size_type>> send_idxs_;

    PartitionInitFunctor(const ExecutorHandler &exec, const label local_size,
                         const label verbose, const int ranks_per_gpu,
                         const CommunicationPattern &communication_pattern)
        : exec_(exec),
          local_size_(local_size),
          verbose_(verbose),
          ranks_per_gpu_(ranks_per_gpu),
          target_ids_(communication_pattern.target_ids),
          target_sizes_(communication_pattern.target_sizes),
          send_idxs_(communication_pattern.send_idxs)
    {}

    void update(std::shared_ptr<part_type> persistent_partition) const
    {
        UNUSED(persistent_partition);
    }

    std::shared_ptr<part_type> init() const
    {
        auto exec = exec_.get_ref_exec();
        auto comm = exec_.get_gko_mpi_host_comm();

        word msg{"initialising partition of size " +
                 std::to_string(local_size_)};
        LOG_1(verbose_, msg)

        if (ranks_per_gpu_ == 1) return {};
    }
};

/* Class handling persistent partitioning information, by default this will
 * store the device partitioning since the host partitioning can easily be
 * reganerated
 *
 * Here device partitioning refers to the partitioning as used on for the Ginkgo
 * data structures and can also reside on the host if the executor is either
 * reference or omp.
 * */
class PersistentPartition
    : public PersistentBase<
          gko::experimental::distributed::Partition<label, label>,
          PartitionInitFunctor> {
    const int ranks_per_gpu_;

    const label local_elements_;

    mutable label global_elements_;

    const ExecutorHandler &exec_;

public:
    using part_type = gko::experimental::distributed::Partition<label, label>;
    /* PersistentPartition constructor using existing memory
     *
     * @param objectRegistry reference to registry for storage
     * @param exec executor handler
     * @param verbose whether to print infos out
     * @param ranks_per_gpu
     * @param offset
     * @param elements
     */
    PersistentPartition(const objectRegistry &db, const ExecutorHandler &exec,
                        const label verbose, const int ranks_per_gpu,
                        const label elements,
                        const CommunicationPattern &communication_pattern)
        : PersistentBase<part_type, PartitionInitFunctor>(
              "device_partition", db,
              PartitionInitFunctor(exec, elements, verbose, ranks_per_gpu,
                                   communication_pattern),
              false, verbose),
          ranks_per_gpu_(ranks_per_gpu),
          local_elements_(elements),
          exec_(exec)
    {
        auto comm = exec.get_gko_mpi_host_comm();
        label local_elements = local_elements_;
        comm->all_reduce(exec.get_ref_exec(), &local_elements, 1, MPI_SUM);
        global_elements_ = local_elements;
    }

    std::shared_ptr<part_type> get_localized_partition() const
    {
        return this->get_persistent_object();
    }

    // number of elements on this rank on the host
    label get_local_size() const { return local_elements_; }

    label get_total_size() const { return global_elements_; }

    label get_ranks_per_gpu() const { return ranks_per_gpu_; }
};

}  // namespace Foam
