// SPDX-License-Identifier: GPLv3 
// SPDX-FileCopyrightText: 2024 OGL authors 

#ifndef OGL_DevicePersistentVector_INCLUDED_H
#define OGL_DevicePersistentVector_INCLUDED_H

#include <functional>

#include <ginkgo/ginkgo.hpp>

#include "DevicePersistent/Base/Base.H"
#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"
#include "DevicePersistent/Partition/Partition.H"
#include "common/common.H"

namespace Foam {


template <class T>
struct VectorInitFunctor {
    using vec = gko::matrix::Dense<scalar>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;

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
    void update(std::shared_ptr<gko::experimental::distributed::Vector<T>>
                    persistent_vector) const
    {
        const auto local_size = partition_.get_local_size();
        word msg{"updating array " + name_ + " of size " +
                 std::to_string(local_size)};
        LOG_1(verbose_, msg)

        init()->move_to(persistent_vector.get());
    }

    std::shared_ptr<gko::experimental::distributed::Vector<T>> init() const
    {
        auto exec =
            (on_device_) ? exec_.get_device_exec() : exec_.get_ref_exec();
        auto comm = exec_.get_gko_mpi_host_comm();

        const auto local_size = partition_.get_local_size();
        word msg{"initialising vector " + name_ + " of size " +
                 std::to_string(local_size)};
        msg += (on_device_) ? " on device" : " on host";
        LOG_1(verbose_, msg)

        auto host_view = gko::array<T>::view(exec_.get_ref_exec(), local_size,
                                             const_cast<T *>(other_));

        if (partition_.get_ranks_per_gpu() == 1) {
            return gko::share(dist_vec::create(
                exec, *exec_.get_gko_mpi_device_comm().get(),
                vec::create(exec, gko::dim<2>{local_size, 1}, host_view, 1)));
        }
    }
};


template <class T>
class PersistentVector
    : public PersistentBase<gko::experimental::distributed::Vector<T>,
                            VectorInitFunctor<T>> {
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using vec = gko::matrix::Dense<scalar>;

    const objectRegistry &db_;

    const word name_;

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
        : PersistentBase<gko::experimental::distributed::Vector<T>,
                         VectorInitFunctor<T>>(
              name, db,
              VectorInitFunctor<T>(exec, name, partition, memory, verbose,
                                   init_on_device),
              update, verbose),
          db_(db),
          name_(name),
          memory_(memory),
          partition_(partition),
          exec_(exec),
          update_(update)
    {}

    /** Copies the content of the distributed vector back to the original source
     **/
    void copy_back()
    {
        const auto local_size = partition_.get_local_size();
        auto global_size = partition_.get_total_size();

        auto comm = exec_.get_gko_mpi_host_comm();
        auto host_device_vector =
            dist_vec::create(exec_.get_ref_exec(), *comm.get());
        host_device_vector->operator=(*get_vector().get());

        // TODO why does it need its size
        // repartitioner size should be sufficient
        auto target_vector = dist_vec::create(exec_.get_ref_exec(), *comm.get(),
                                              gko::dim<2>{global_size, 1},
                                              gko::dim<2>{local_size, 1}, 1);
        scalar *host_buffer = host_device_vector->get_local_values();

        auto host_buffer_view =
            gko::array<T>::view(exec_.get_ref_exec(), local_size, host_buffer);

        auto to_view = gko::array<T>::view(exec_.get_ref_exec(), local_size,
                                           const_cast<T *>(memory_));
        to_view = host_buffer_view;
    }

    /** Writes the content of the distributed vector to disk
     **
     ** Data is stored as .mtx file under processor?/<time>/<name_>.mtx
     **/
    void write() const
    {
        export_vec(name_, get_vector()->get_local_vector(), db_);
    }

    // getter and setter

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

    const ExecutorHandler &get_exec_handler() const { return exec_; }

    std::shared_ptr<gko::experimental::distributed::Vector<T>> get_vector()
        const
    {
        return this->get_persistent_object();
    }
};

}  // namespace Foam

#endif
