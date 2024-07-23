// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later


#pragma once

#include <functional>

#include <ginkgo/ginkgo.hpp>

#include "DevicePersistent/Base/Base.H"
#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"
#include "common/common.H"

namespace Foam {


template <class T>
struct ArrayInitFunctor {
    const word name_;

    const ExecutorHandler &exec_;

    const label size_;

    const label verbose_;

    const bool on_device_;

    // Memory from which array will be initialised
    const T *other_;


    ArrayInitFunctor(const ExecutorHandler &exec, const word name,
                     const label size, const label verbose,
                     const bool on_device)
        : exec_(exec),
          name_(name),
          size_(size),
          on_device_(on_device),
          verbose_(verbose),
          other_(NULL)
    {}

    ArrayInitFunctor(const ExecutorHandler &exec, const T *other,
                     const word name, const label size, const label verbose,
                     const bool on_device)
        : exec_(exec),
          name_(name),
          size_(size),
          on_device_(on_device),
          verbose_(verbose),
          other_(other)
    {}


    // update persistent array from host memory
    void update(std::shared_ptr<gko::array<T>> persistent_array) const
    {
        if (other_ != NULL) {
            word msg{"updating array " + name_ + " of size " +
                     std::to_string(size_)};
            LOG_1(verbose_, msg)

            auto host_view = gko::array<T>::view(exec_.get_ref_exec(), size_,
                                                 const_cast<T *>(other_));
            persistent_array->operator=(host_view);
        }
    }

    std::shared_ptr<gko::array<T>> init() const
    {
        auto exec =
            (on_device_) ? exec_.get_device_exec() : exec_.get_ref_exec();
        word msg{"initialising array " + name_ + " of size " +
                 std::to_string(size_)};
        msg += (on_device_) ? " on device" : " on host";
        LOG_1(verbose_, msg)

        if (other_ != NULL) {
            auto host_view = gko::array<T>::view(exec_.get_ref_exec(), size_,
                                                 const_cast<T *>(other_));
            return std::make_shared<gko::array<T>>(exec, host_view);
        } else {
            return std::make_shared<gko::array<T>>(exec, size_);
        }
    }
};

template <class T>
class PersistentArray
    : public PersistentBase<gko::array<T>, ArrayInitFunctor<T>> {
    using vec = gko::matrix::Dense<scalar>;

    T *memory_;

    mutable label size_;

    // indicating if the underlying array needs to
    // updated even if was found in the object registry
    const bool update_;

    // The same as dpb but for the global data
    // NOTE if on the host:
    //  - This should exist with the full length only on the master
    // on the device:
    //  - global data is needed on the device for example for the
    //    col and row idxs
    mutable std::shared_ptr<gko::matrix::Dense<T>> persistent_object_global_{};


public:
    /* PersistentArray constructor
     *
     * @param name name of the underlying field or data
     * @param objectRegistry reference to registry for storage
     * @param exec executor handler
     * @param verbose whether to print infos out
     * @param update whether to update the underlying array if found in registry
     * TODO fix documentation
     * @param init_on_device whether to update the underlying array if found in
     * registry
     */
    PersistentArray(const word name, const objectRegistry &db,
                    const ExecutorHandler &exec, const label size,
                    const label verbose, const bool update,
                    const bool on_device)
        : PersistentBase<gko::array<T>, ArrayInitFunctor<T>>(
              name, db,
              ArrayInitFunctor<T>(exec, name, size, verbose, on_device), update,
              verbose),
          size_(size),
          update_(update)
    {}

    /* PersistentArray constructor using existing memory
     *
     * @param memory ptr to memory on host from which the gko array is
     *               initialized
     * @param name name of the underlying field or data
     * @param objectRegistry reference to registry for storage
     * @param exec executor handler
     * @param verbose whether to print infos out
     * @param update whether to update the underlying array if found in registry
     * @param init_on_device whether the array is to be initialized on the
     * device or host
     */
    PersistentArray(T *memory, const word name, const objectRegistry &db,
                    const ExecutorHandler &exec, const label size,
                    const label verbose, const bool update,
                    const bool init_on_device)
        : PersistentBase<gko::array<T>, ArrayInitFunctor<T>>(
              name, db,
              ArrayInitFunctor<T>(exec, memory, name, size, verbose,
                                  init_on_device),
              update, verbose),
          memory_(memory),
          size_(size),
          update_(update)
    {}

    bool get_update() const { return update_; }

    T *get_data() const { return this->get_persistent_object()->get_data(); }

    void set_data(T *data) { this->get_persistent_object()->get_data() = data; }

    const T *get_const_data() const
    {
        return this->get_persistent_object()->get_const_data();
    }


    std::shared_ptr<gko::array<T>> get_array() const
    {
        return this->get_persistent_object();
    }

    /* Create a Dense<T> vector from underlying data
     * */
    // std::shared_ptr<gko::matrix::Dense<T>> get_dense_vec() const
    // {
    //     auto array = get_array();
    //     auto result = gko::share(vec::create(
    //         exec_.get_device_exec(), gko::dim<2>(array->get_num_elems(), 1),
    //         gko::array<T>::view(exec_.get_device_exec(), array->get_num_elems(),
    //                             get_data()),
    //         1));
    //     return result;
    // }

    /* Copies from a distributed::Vector<T> back into raw host memory
     *
     * TODO store and use *memory
     * */
    void copy_back(
        const std::shared_ptr<gko::experimental::distributed::Vector<T>> dist_x)
    {
        auto dist_device_exec = dist_x->get_executor();
        auto host_view = gko::array<T>::view(dist_device_exec->get_master(), size_,
                                             const_cast<T *>(memory_));

        auto x_view = gko::array<T>::view(dist_device_exec, size_,
                                          dist_x->get_local()->get_values());

        host_view = x_view;
    }

    // void copy_back(const std::shared_ptr<gko::matrix::Dense<T>> device_x)
    // {
    //     auto host_view = gko::array<T>::view(exec_.get_ref_exec(), size_,
    //                                          const_cast<T *>(memory_));
    //
    //     auto x_view = gko::array<T>::view(exec_.get_device_exec(), size_,
    //                                       device_x->get_values());
    //
    //     host_view = x_view;
    // }

    void set_size(const label size) const { size_ = size; }
};

}  // namespace Foam
