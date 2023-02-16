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

Class
    Foam::DevicePersistentPartition

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    DevicePersistentPartition.H

\*---------------------------------------------------------------------------*/
#ifndef OGL_DevicePersistentPartition_INCLUDED_H
#define OGL_DevicePersistentPartition_INCLUDED_H

#include <functional>

#include <ginkgo/ginkgo.hpp>
#include "../DevicePersistentBase/DevicePersistentBase.H"
#include "../ExecutorHandler/ExecutorHandler.H"

#include "../common/common.H"

namespace Foam {


struct PartitionInitFunctor {
    using vec = gko::matrix::Dense<scalar>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    const ExecutorHandler &exec_;

    const label rows_end_;

    const label rows_start_;

    const label verbose_;

    const int ranks_per_gpu_;


    PartitionInitFunctor(const ExecutorHandler &exec, const label rows_start,
                         const label rows_end, const label verbose,
                         const int ranks_per_gpu)
        : exec_(exec),
          rows_end_(rows_end),
          rows_start_(rows_start),
          verbose_(verbose),
          ranks_per_gpu_(ranks_per_gpu)
    {}

    void update(
        std::shared_ptr<gko::experimental::distributed::Partition<label, label>>
            persistent_partition) const
    {}

    std::shared_ptr<gko::experimental::distributed::Partition<label, label>>
    init() const
    {
        auto exec = exec_.get_ref_exec();
        auto comm = exec_.get_gko_mpi_host_comm();

        word msg{"initialising partition of size " +
                 std::to_string(rows_end_ - rows_start_)};
        LOG_1(verbose_, msg)

        auto host_partition = gko::share(
            gko::experimental::distributed::build_partition_from_local_range<
                label, label>(exec, *comm.get(),
                              gko::span(rows_start_, rows_end_)));

        if (ranks_per_gpu_ == 1) return host_partition;

        auto local_rank = comm->rank();
        auto local_owning_rank = ranks_per_gpu_ * (local_rank / ranks_per_gpu_);
        auto next_owning_rank = ranks_per_gpu_ + local_owning_rank;

        // The new partition ranges up to next owning rank elements
        auto target_rows_end =
            host_partition->get_range_bounds()[next_owning_rank];

        // if we are not on a owning rank start rows are set to be the end
        // row else the rows start are unchanged
        auto target_rows_start =
            (local_rank % ranks_per_gpu_ != 0) ? target_rows_end : rows_start_;
        return gko::share(
            gko::experimental::distributed::build_partition_from_local_range<
                label, label>(exec, *comm.get(),
                              gko::span(target_rows_start, target_rows_end)));
    }
};

/* Class handling persistent partitioning information, by default this will
 * store the device partitioning since the host partitioning can easily be
 * reganerated
 *
 * Here device partitioning refers to the partitioning as used on for the Ginkgo
 * data structures and can also reside on the host if the executor is either
 * reference or omp.
 *
 *
 *
 * */
class PersistentPartition
    : public PersistentBase<
          gko::experimental::distributed::Partition<label, label>,
          PartitionInitFunctor> {
    const int ranks_per_gpu_;

    const label offset_;

    const label elements_;

    const ExecutorHandler &exec_;


public:
    /* PersistentPartition constructor using existing memory
     *
     * @param objectRegistry reference to registry for storage
     * @param exec executor handler
     * @param verbose whether to print infos out
     * device or host
     * @param ranks_per_gpu
     */
    PersistentPartition(const objectRegistry &db, const ExecutorHandler &exec,
                        const label verbose, const int ranks_per_gpu,
                        const label offset, const label elements)
        : PersistentBase<
              gko::experimental::distributed::Partition<label, label>,
              PartitionInitFunctor>(
              "device_partition", db,
              PartitionInitFunctor(exec, offset, offset + elements, verbose,
                                   ranks_per_gpu),
              false, verbose),
          ranks_per_gpu_(ranks_per_gpu),
          offset_(offset),
          elements_(elements),
          exec_(exec)
    {}

    std::shared_ptr<gko::experimental::distributed::Partition<label, label>>
    get_device_partition() const
    {
        return this->get_persistent_object();
    }

    std::shared_ptr<
        const gko::experimental::distributed::Partition<label, label>>
    get_host_partition() const
    {
        auto comm = exec_.get_gko_mpi_host_comm();
        return gko::share(
            gko::experimental::distributed::build_partition_from_local_range<
                label, label>(exec_.get_ref_exec(), *comm.get(),
                              gko::span(offset_, offset_ + elements_)));
    }

    // number of elements on this rank on the host
    label get_local_host_size() const { return elements_; }

    // number of elements on this rank on the device
    label get_local_device_size() const
    {
        auto comm = exec_.get_gko_mpi_device_comm();
        return this->get_persistent_object()->get_part_size(comm->rank());
    }


    label get_total_size() const
    {
        return this->get_persistent_object()->get_size();
    }

    label get_ranks_per_gpu() const { return ranks_per_gpu_; }
};

}  // namespace Foam

#endif
