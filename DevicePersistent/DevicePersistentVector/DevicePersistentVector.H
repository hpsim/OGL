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
    Foam::DevicePersistentVector

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    DevicePersistentVector.H

\*---------------------------------------------------------------------------*/
#ifndef OGL_DevicePersistentVector_INCLUDED_H
#define OGL_DevicePersistentVector_INCLUDED_H

#include <functional>

#include <ginkgo/ginkgo.hpp>
#include "../DevicePersistentBase/DevicePersistentBase.H"
#include "../DevicePersistentPartition/DevicePersistentPartition.H"
#include "../ExecutorHandler/ExecutorHandler.H"

#include "../common/common.H"

namespace Foam {


template <class T>
struct VectorInitFunctor {
    using vec = gko::matrix::Dense<scalar>;
    using dist_vec = gko::distributed::Vector<scalar>;

    const word name_;

    const ExecutorHandler &exec_;

    // representing the partitioning that OF uses
    const PersistentPartition &partition_;

    const label verbose_;

    const bool on_device_;

    // Memory from which array will be initialised
    const T *other_;


    VectorInitFunctor(const ExecutorHandler &exec, const word name,
                      const PersistentPartition &partition, const T *other,
                      const label verbose, const bool on_device = false)
        : exec_(exec),
          name_(name),
          partition_(partition),
          on_device_(on_device),
          verbose_(verbose),
          other_(other)
    {}


    // update persistent array from host memory
    void update(
        std::shared_ptr<gko::distributed::Vector<T>> persistent_vector) const
    {
        // auto comm = exec_.get_gko_mpi_comm_wrapper();
        // const auto local_size = partition_->get_part_size(comm->rank());
        // word msg{"updating array " + name_ + " of size " +
        //          std::to_string(local_size)};
        // LOG_1(verbose_, msg)

        // auto host_exec = exec_.get_ref_exec();
        // auto device_exec = exec_.get_device_exec();
        // auto exec =
        //     (on_device_) ? exec_.get_device_exec() : exec_.get_ref_exec();

        // host_exec->copy_from(exec.get(), local_size, other_,
        //                      persistent_vector->get_local_values());
    }

    std::shared_ptr<gko::distributed::Vector<T>> init() const
    {
        auto exec =
            (on_device_) ? exec_.get_device_exec() : exec_.get_ref_exec();
        auto comm = exec_.get_gko_mpi_host_comm();

        const auto local_size = partition_.get_local_host_size();
        word msg{"initialising vector " + name_ + " of size " +
                 std::to_string(local_size)};
        msg += (on_device_) ? " on device" : " on host";
        LOG_1(verbose_, msg)

        auto host_view = gko::array<T>::view(exec_.get_ref_exec(), local_size,
                                             const_cast<T *>(other_));

        if (partition_.get_ranks_per_gpu() == 1) {
            return gko::share(dist_vec::create(
                *exec_.get_gko_mpi_host_comm().get(),
                vec::create(exec, gko::dim<2>{local_size, 1}, host_view, 1)
                    .get()));
        }

        // when having more ranks than GPUs we must first repartition and
        // gather the vector
        auto host_vector = gko::share(dist_vec::create(
            *exec_.get_gko_mpi_host_comm().get(),
            vec::create(exec, gko::dim<2>{local_size, 1}, host_view, 1).get()));

        auto target_vector = gko::share(dist_vec::create(
            *exec_.get_gko_mpi_host_comm().get(),
            vec::create(exec, gko::dim<2>{local_size, 1}, host_view, 1).get()));

        // TODO test if this needs to be persistent
        auto repartitioner =
            gko::share(gko::distributed::repartitioner<label, label>::create(
                *comm.get(), partition_.get_host_partition(),
                partition_.get_device_partition()));

        repartitioner->gather(host_vector.get(), target_vector.get());

        auto device_vec = gko::share(
            dist_vec::create(*exec_.get_gko_mpi_device_comm().get()));

        target_vector->move_to(device_vec.get());

        return device_vec;
    }
};


template <class T>
class PersistentVector
    : public PersistentBase<gko::distributed::Vector<T>, VectorInitFunctor<T>> {
    using dist_vec = gko::distributed::Vector<scalar>;
    using vec = gko::matrix::Dense<scalar>;

    const T *memory_;

    const PersistentPartition partition_;

    const ExecutorHandler &exec_;

    // indicating if the underlying array needs to
    // updated even if was found in the object registry
    const bool update_;


public:
    /* PersistentVector constructor using existing memory
     *
     * @param memory ptr to memory on host from which the gko array is
     *               initialized
     * @param name name of the underlying field or data
     * @param objectRegistry reference to registry for storage
     * @param exec executor handler
     * @param partition Only needed to compute local and global size
     * @param verbose whether to print infos out
     * @param update whether to update the underlying array if found in registry
     * @param init_on_device whether the array is to be initialized on the
     * device or host
     * @param ranks_per_gpu
     */
    PersistentVector(const T *memory, const word name, const objectRegistry &db,
                     const ExecutorHandler &exec,
                     const PersistentPartition partition, const label verbose,
                     const bool update, const bool init_on_device)
        : PersistentBase<gko::distributed::Vector<T>, VectorInitFunctor<T>>(
              name, db,
              VectorInitFunctor<T>(exec, name, partition, memory, verbose,
                                   init_on_device),
              update, verbose),
          memory_(memory),
          partition_(partition),
          exec_(exec),
          update_(update)
    {}

    // label get_global_size() const { return partition_.size(); }

    bool get_update() const { return update_; }

    T *get_data() const { return this->get_persistent_object()->get_data(); }

    void set_data(T *data)
    {
        this->get_persistent_object()->get_data() = data;
    };

    const T *get_const_data() const
    {
        return this->get_persistent_object()->get_const_data();
    };


    std::shared_ptr<gko::distributed::Vector<T>> get_vector() const
    {
        return this->get_persistent_object();
    }

    void copy_back()
    {
        const auto local_device_size = partition_.get_local_device_size();

        auto comm = exec_.get_gko_mpi_host_comm();
        auto host_device_vector = dist_vec::create(*comm.get());
        host_device_vector->operator=(*get_vector().get());

        auto target_vector = dist_vec::create(*comm.get());
        scalar *host_buffer = host_device_vector->get_local_values();

        if (partition_.get_ranks_per_gpu() != 1) {

            auto repartitioner = gko::share(
                gko::distributed::repartitioner<label, label>::create(
                    *comm.get(), partition_.get_host_partition(),
                    partition_.get_device_partition()));

            repartitioner->scatter(target_vector.get(),
                                   host_device_vector.get());
        }

        auto local_size = partition_.get_local_host_size();
        auto host_buffer_view =
            gko::array<T>::view(exec_.get_ref_exec(), local_size, host_buffer);

        auto to_view = gko::array<T>::view(exec_.get_ref_exec(), local_size,
                                           const_cast<T *>(memory_));
        to_view = host_buffer_view;
    }

    const ExecutorHandler &get_exec_handler() const { return exec_; }
};

}  // namespace Foam

#endif
