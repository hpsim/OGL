// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>

#include <ginkgo/ginkgo.hpp>

#include "DevicePersistent/Base/Base.H"
#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"
#include "DevicePersistent/Partition/Partition.H"
#include "MatrixWrapper/CommunicationPattern/CommunicationPattern.H"
#include "MatrixWrapper/LDUMatrix/HostMatrix.H"
#include "Repartitioner/Repartitioner.H"
#include "common/common.H"

namespace Foam {


template <class T>
struct VectorInitFunctor {
    using vec = gko::matrix::Dense<scalar>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    const word name_;

    const Repartitioner &repartitioner_;

    const ExecutorHandler &exec_;

    const label verbose_;

    const bool on_device_;

    // Memory from which array will be initialised
    const T *other_;

    const label size_;

    VectorInitFunctor(const ExecutorHandler &exec, const word name,
                      const Repartitioner &repartitioner, const T *other,
                      const label size, const label verbose,
                      const bool on_device = false)
        : name_(name),
          repartitioner_(repartitioner),
          exec_(exec),
          on_device_(on_device),
          verbose_(verbose),
          other_(other),
          size_(size)
    {}


    // update persistent array from host memory
    void update(std::shared_ptr<gko::experimental::distributed::Vector<T>>
                    persistent_vector) const
    {
        const auto local_size = size_;
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
        auto rank = comm->rank();

        const auto local_size = size_;
        label repart_size = repartitioner_.get_repart_size();
        word msg{"initialising vector " + name_ + " of size " +
                 std::to_string(repart_size)};
        msg += (on_device_) ? " on device" : " on host";
        LOG_1(verbose_, msg)

        auto host_view =
            gko::array<T>::const_view(exec_.get_ref_exec(), local_size, other_);

        // FIXME this shouldn't be needed here also not during an
        // update
        auto comm_pattern = compute_send_recv_counts(
            exec_, repartitioner_.get_ranks_per_gpu(), local_size);

        // this should happen on device
        auto local_coeffs = gko::array<scalar>(exec, repart_size);

        communicate_values(exec_, comm_pattern, host_view.get_const_data(),
                           local_coeffs.get_data());

        auto ret = gko::share(dist_vec::create(
            exec, *comm.get(),
            vec::create(exec, gko::dim<2>{repart_size, 1}, local_coeffs, 1)));
        return ret;
    }
};


template <class T>
class PersistentVector
    : public PersistentBase<gko::experimental::distributed::Vector<T>,
                            VectorInitFunctor<T>> {
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using part_type = gko::experimental::distributed::Partition<label, label>;
    using vec = gko::matrix::Dense<scalar>;

    const objectRegistry &db_;

    const word name_;

    const T *memory_;

    const label size_;

    const ExecutorHandler &exec_;

    // indicating if the underlying array needs to
    // updated even if was found in the object registry
    const bool update_;

    const Repartitioner & repartitioner_;

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
    PersistentVector(const T *memory, const label size, const word name,
                     const Repartitioner &repartitioner,
                     const objectRegistry &db, const ExecutorHandler &exec,
                     const label verbose, const bool update,
                     const bool init_on_device)
        : PersistentBase<gko::experimental::distributed::Vector<T>,
                         VectorInitFunctor<T>>(
              name, db,
              VectorInitFunctor<T>(exec, name, repartitioner, memory, size,
                                   verbose, init_on_device),
              update, verbose),
          db_(db),
          name_(name),
          memory_(memory),
          size_(size),
          exec_(exec),
          update_(update),
          repartitioner_(repartitioner)
    {}

    /** Copies the content of the distributed vector back to the original source
     **/
    void copy_back()
    {
        auto comm_pattern = compute_scatter_counts(
            exec_, repartitioner_.get_ranks_per_gpu(), size_);

        // std::cout << __FILE__ << ":" << __LINE__ << " rank " << repartitioner_.get_rank()
        //     << " comm_pattern send " << std::get<0>(comm_pattern)
        //     << " comm_pattern recv " << std::get<1>(comm_pattern)
        //     << " comm_pattern send_offs " << std::get<2>(comm_pattern)
        //     << " comm_pattern recv_offs " << std::get<3>(comm_pattern)
        //     << "\n";

        communicate_values(exec_, comm_pattern,
                           get_vector()->get_local_values(),
                           const_cast<T *>(memory_));
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
