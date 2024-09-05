// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>

#include <ginkgo/ginkgo.hpp>

#include "OGL/DevicePersistent/Base.H"
#include "OGL/DevicePersistent/ExecutorHandler.H"
#include "OGL/MatrixWrapper/HostMatrix.H"
#include "OGL/common.H"

namespace Foam {


struct PartitionInitFunctor {
    using comm_size_type = int;
    using part_type = gko::experimental::distributed::Partition<label, label>;

    const ExecutorHandler &exec_;

    const label local_size_;

    const label verbose_;

    PartitionInitFunctor(const ExecutorHandler &exec, const label verbose,
                         const int ranks_per_gpu,
                         std::shared_ptr<const HostMatrixWrapper> host_matrix)
        : exec_(exec),
          local_size_(host_matrix->get_local_nrows()),
          verbose_(verbose)

              void update(std::shared_ptr<part_type> persistent_partition) const
    {
        UNUSED(persistent_partition);
    }

    std::shared_ptr<part_type> init() const
    {
        word msg{"initialising partition of size " +
                 std::to_string(local_size_)};
        LOG_1(verbose_, msg)
        auto exec = exec_.get_ref_exec();
        auto comm = exec_.get_gko_mpi_host_comm();
        return build_partition_from_local_size<label, label>(ref_exec, comm,
                                                             local_size_);
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
                        std::shared_ptr<const HostMatrixWrapper> host_matrix)
        : PersistentBase<part_type, PartitionInitFunctor>(
              "device_partition", db,
              PartitionInitFunctor(exec, verbose, ranks_per_gpu, host_matrix),
              false, verbose),
          ranks_per_gpu_(ranks_per_gpu),
          local_elements_(elements)
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
