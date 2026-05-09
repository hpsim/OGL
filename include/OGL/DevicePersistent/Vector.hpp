// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>

#include <ginkgo/ginkgo.hpp>

#include "OGL/DevicePersistent/Base.hpp"
#include "OGL/DevicePersistent/ExecutorHandler.hpp"
#include "OGL/Repartitioner.hpp"
#include "OGL/common.hpp"

namespace Foam {


template <class T>
struct VectorInitFunctor {
    using vec = gko::matrix::Dense<scalar>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    const ExecutorHandler &exec_;

    const word name_;

    const label verbose_;

    std::shared_ptr<const RepartDistMatrix> dist_matrix_;

    const bool on_device_;

    // Memory from which array will be initialised
    const T *other_;

    VectorInitFunctor(const ExecutorHandler &exec, const word name,
                      std::shared_ptr<const RepartDistMatrix> dist_matrix,
                      const T *other, const label verbose,
                      const bool on_device = false)
        : exec_(exec),
          name_(name),
          verbose_(verbose),
          dist_matrix_(dist_matrix),
          on_device_(on_device),
          other_(other)
    {}


    // update persistent array from host memory
    void update(std::shared_ptr<gko::experimental::distributed::Vector<T>>
                    persistent_vector) const
    {
        auto ref_exec = exec_.get_ref_exec();
        auto exec = exec_.get_device_exec();
        auto comm = exec_.get_host_comm();
        auto repartitioner = dist_matrix_->get_repartitioner();
        auto host_size = repartitioner->get_orig_size();
        auto repart_size = repartitioner->get_repart_size();
        word msg{"updating array " + name_ + " of host size " +
                 std::to_string(host_size) + " repartitioned size " +
                 std::to_string(repart_size)};
        LOG_1(verbose_, msg)

        auto host_view = gko::array<T>::const_view(ref_exec, host_size, other_);

        //// TODO store
        auto comm_pattern = compute_gather_to_owner_counts(
            exec_, repartitioner->get_ranks_per_gpu(), host_size);
        bool host_buffer = exec_.get_gko_force_host_buffer();

        communicate_values(ref_exec, exec, comm, comm_pattern,
                           host_view.get_const_data(),
                           persistent_vector->get_local_values(), host_buffer);
    }

    std::shared_ptr<gko::experimental::distributed::Vector<T>> init() const
    {
        auto exec = exec_.get_device_exec();
        auto ref_exec = exec_.get_ref_exec();
        auto repartitioner = dist_matrix_->get_repartitioner();
        auto host_size = repartitioner->get_orig_size();
        auto repart_size = repartitioner->get_repart_size();

        word msg{"initialising vector " + name_ + " of size " +
                 std::to_string(repart_size) + " orig size " +
                 std::to_string(host_size)};
        LOG_1(verbose_, msg)

        auto host_view =
            gko::array<T>::const_view(exec_.get_ref_exec(), host_size, other_);

        // TODO store, this could be stored in dist_matrix_
        auto comm_pattern = compute_gather_to_owner_counts(
            exec_, repartitioner->get_ranks_per_gpu(), host_size);

        auto values = gko::array<scalar>(ref_exec, repart_size);

        communicate_values(exec_, comm_pattern, host_view.get_const_data(),
                           values.get_data());

        auto device_comm = exec_.get_device_comm();
        auto ret = gko::share(dist_vec::create(
            exec, *device_comm.get(),
            vec::create(
                exec, gko::dim<2>{static_cast<gko::size_type>(repart_size), 1},
                values, 1)));

        word done_msg{"done initialising vector " + name_};
        LOG_1(verbose_, done_msg)
        return ret;
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

    std::shared_ptr<const RepartDistMatrix> dist_matrix_;

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
                     std::shared_ptr<const RepartDistMatrix> dist_matrix,
                     const label verbose, const bool update,
                     const bool init_on_device)
        : PersistentBase<gko::experimental::distributed::Vector<T>,
                         VectorInitFunctor<T>>(
              name, db,
              VectorInitFunctor<T>(exec, name, dist_matrix, memory, verbose,
                                   init_on_device),
              update, verbose),
          db_(db),
          name_(name),
          memory_(memory),
          dist_matrix_(dist_matrix),
          exec_(exec),
          update_(update)
    {}

    /** Copies the content of the distributed vector back to the original source
     **/
    void copy_back()
    {
        auto exec = exec_.get_device_exec();
        auto ref_exec = exec_.get_ref_exec();
        auto repart_comm = exec_.get_repart_comm();

        auto repartitioner = dist_matrix_->get_repartitioner();
        auto host_size = repartitioner->get_orig_size();

        auto comm_pattern = compute_scatter_from_owner_counts(
            exec_, repartitioner->get_ranks_per_gpu(), host_size);

        label owner_rank = exec_.get_owner_rank();
        auto repartAllToAll =
            compute_repart_allToall(exec_, comm_pattern, owner_rank);

        // communicate_values(exec, ref_exec, comm, comm_pattern,
        //                    get_vector()->get_local_values(),
        //                    const_cast<T *>(memory_), host_buffer);

        label send_size = comm_pattern.send_offsets.back();
        auto send_view = gko::array<scalar>::const_view(
            exec, send_size, get_vector()->get_local_values());
        auto tmp = gko::array<scalar>(exec, send_size);

        tmp = send_view;
        tmp.set_executor(ref_exec);

        MPI_Request copy_back_req;
        MPI_Iscatterv(tmp.get_data(), repartAllToAll.send_counts.data(),
                      repartAllToAll.send_offsets.data(), MPI_DOUBLE,
                      const_cast<T *>(memory_), repartAllToAll.recv_counts[0],
                      MPI_DOUBLE, 0, repart_comm->get(), &copy_back_req);
        MPI_Wait(&copy_back_req, MPI_STATUS_IGNORE);
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

    void set_data(T *data) { this->get_persistent_object()->get_data() = data; }

    const T *get_const_data() const
    {
        return this->get_persistent_object()->get_const_data();
    }

    const ExecutorHandler &get_exec_handler() const { return exec_; }

    std::shared_ptr<gko::experimental::distributed::Vector<T>> get_vector()
        const
    {
        return this->get_persistent_object();
    }
};

}  // namespace Foam
