// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>

#include <ginkgo/ginkgo.hpp>

#include "OGL/DevicePersistent/Base.H"
#include "OGL/DevicePersistent/ExecutorHandler.H"
#include "OGL/Repartitioner.H"
#include "OGL/common.H"

namespace Foam {


template <class T>
struct VectorInitFunctor {
    using vec = gko::matrix::Dense<scalar>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    const word name_;

    const ExecutorHandler &exec_;

    std::shared_ptr<const RepartDistMatrix> dist_matrix_;

    const label verbose_;

    const bool on_device_;

    // Memory from which array will be initialised
    const T *other_;

    VectorInitFunctor(const ExecutorHandler &exec, const word name,
                      std::shared_ptr<const RepartDistMatrix> dist_matrix,
                      const T *other, const label verbose,
                      const bool on_device = false)
        : exec_(exec),
          name_(name),
          dist_matrix_(dist_matrix),
          on_device_(on_device),
          verbose_(verbose),
          other_(other)
    {}


    // update persistent array from host memory
    void update(std::shared_ptr<gko::experimental::distributed::Vector<T>>
                    persistent_vector) const
    {
        auto repartitioner = dist_matrix_->get_repartitioner();
        auto host_size = repartitioner->get_orig_size();
        auto repart_size = repartitioner->get_repart_size();
        word msg{"updating array " + name_ + " of host size " +
                 std::to_string(host_size) + " repartitioned size " +
                 std::to_string(repart_size)};
        LOG_1(verbose_, msg)

        auto ref_exec = exec_.get_ref_exec();
        auto host_view = gko::array<T>::const_view(ref_exec, host_size, other_);

        // TODO store
        auto comm_pattern = compute_gather_to_owner_counts(
            exec_, repartitioner->get_ranks_per_gpu(), host_size);

        communicate_values(exec_, comm_pattern, host_view.get_const_data(),
                           persistent_vector->get_local_values());
    }

    std::shared_ptr<gko::experimental::distributed::Vector<T>> init() const
    {
        auto exec = exec_.get_device_exec();
        auto ref_exec = exec_.get_ref_exec();
        auto comm = exec_.get_communicator();
        auto repartitioner = dist_matrix_->get_repartitioner();
        auto host_size = repartitioner->get_orig_size();
        auto repart_size = repartitioner->get_repart_size();
    std::cout<<__FILE__<< ":" << __LINE__ <<"vector\n";

        word msg{"initialising vector " + name_ + " of size " +
                 std::to_string(repart_size) + " orig size " +
                 std::to_string(host_size)};
        LOG_1(verbose_, msg)

    std::cout<<__FILE__<< ":" << __LINE__ <<"vector\n";
        auto host_view =
            gko::array<T>::const_view(exec_.get_ref_exec(), host_size, other_);

    std::cout<<__FILE__<< ":" << __LINE__ <<"vector\n";
        // TODO store
        auto comm_pattern = compute_gather_to_owner_counts(
            exec_, repartitioner->get_ranks_per_gpu(), host_size);

        auto values = gko::array<scalar>(ref_exec, repart_size);

    std::cout<<__FILE__<< ":" << __LINE__ <<"vector\n";
//       communicate_values(exec_, comm_pattern, host_view.get_const_data(),
//                          values.get_data());
    comm->all_to_all_v(ref_exec, host_view.get_const_data(), comm_pattern.send_counts.data(),
                      comm_pattern.send_offsets.data(), values.get_data(),
                      comm_pattern.recv_counts.data(),
                      comm_pattern.recv_offsets.data());

    std::cout<<__FILE__<< ":" << __LINE__ <<"vector\n";
        auto ret = gko::share(dist_vec::create(
            exec, *comm.get(),
            vec::create(exec, gko::dim<2>{repart_size, 1}, values, 1)));

    std::cout<<__FILE__<< ":" << __LINE__ <<"vector\n";
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
        auto comm = exec_.get_communicator();

        std::cout<<__FILE__<< ":" << __LINE__ <<"copy vector\n";
        auto repartitioner = dist_matrix_->get_repartitioner();
        auto host_size = repartitioner->get_orig_size();

        std::cout<<__FILE__<< ":" << __LINE__ <<"copy vector\n";
        auto comm_pattern = compute_scatter_from_owner_counts(
            exec_, repartitioner->get_ranks_per_gpu(), host_size);

         std::vector<scalar> host_send_buffer;
         scalar *send_ptr;

      std::cout<<__FILE__<< ":" << __LINE__ <<"copy back vector\n";
      communicate_values(exec, ref_exec, comm,  comm_pattern,
		      get_vector()->get_local_values(),
                         const_cast<T *>(memory_), true);
      std::cout<<__FILE__<< ":" << __LINE__ <<"done copy vector\n";
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
