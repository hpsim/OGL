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
    Foam::DevicePersistentArray

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    DevicePersistentArray.H

\*---------------------------------------------------------------------------*/
#ifndef OGL_DevicePersistentArray_INCLUDED_H
#define OGL_DevicePersistentArray_INCLUDED_H

#include <functional>

#include <ginkgo/ginkgo.hpp>
#include "../DevicePersistentBase/DevicePersistentBase.H"
#include "../ExecutorHandler/ExecutorHandler.H"
#include "../IOGlobalIndex/gkoGlobalIndex.H"

#include "../common/common.H"

namespace Foam {


template <class T>
struct ArrayInitFunctor {
    const word name_;

    const ExecutorHandler &exec_;

    const gkoGlobalIndex &global_index_;

    const label verbose_;

    const bool on_device_;

    // Memory from which array will be initialised
    const T *other_;


    ArrayInitFunctor(const ExecutorHandler &exec, const word name,
                     const gkoGlobalIndex &global_index, const label verbose,
                     const bool on_device = false)
        : exec_(exec),
          name_(name),
          global_index_(global_index),
          on_device_(on_device),
          verbose_(verbose),
          other_(NULL)
    {}

    ArrayInitFunctor(const ExecutorHandler &exec, const word name,
                     const gkoGlobalIndex &global_index, const T *other,
                     const label verbose, const bool on_device = false)
        : exec_(exec),
          name_(name),
          global_index_(global_index),
          on_device_(on_device),
          verbose_(verbose),
          other_(other)
    {}


    // update persistent array from host memory
    void update(std::shared_ptr<gko::array<T>> persistent_array) const
    {
        if (other_ != NULL) {
            word msg{"updating array " + name_ + " of size " +
                     std::to_string(global_index_.localSize())};
            LOG_1(verbose_, msg)

            auto host_view = gko::array<T>::view(exec_.get_ref_exec(),
                                                 global_index_.localSize(),
                                                 const_cast<T *>(other_));
            persistent_array->operator=(host_view);
        }
    }

    std::shared_ptr<gko::array<T>> init() const
    {
        auto exec =
            (on_device_) ? exec_.get_device_exec() : exec_.get_ref_exec();
        word msg{"initialising array " + name_ + " of size " +
                 std::to_string(global_index_.localSize())};
        msg += (on_device_) ? " on device" : " on host";
        LOG_1(verbose_, msg)

        if (other_ != NULL) {
            auto host_view = gko::array<T>::view(exec_.get_ref_exec(),
                                                 global_index_.localSize(),
                                                 const_cast<T *>(other_));
            return std::make_shared<gko::array<T>>(exec, host_view);
        } else {
            return std::make_shared<gko::array<T>>(exec,
                                                   global_index_.localSize());
        }
    }
};

template <class T>
class PersistentArray
    : public PersistentBase<gko::array<T>, ArrayInitFunctor<T>> {
    using vec = gko::matrix::Dense<scalar>;

    T *memory_;

    const gkoGlobalIndex &global_index_;

    const ExecutorHandler &exec_;

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
                    const ExecutorHandler &exec,
                    const gkoGlobalIndex &global_index, const label verbose,
                    const bool update, const bool on_device)
        : PersistentBase<gko::array<T>, ArrayInitFunctor<T>>(
              name, db,
              ArrayInitFunctor<T>(exec, name, global_index, verbose, on_device),
              update, verbose),
          global_index_(global_index),
          exec_(exec),
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
                    const ExecutorHandler &exec,
                    const gkoGlobalIndex &global_index, const label verbose,
                    const bool update, const bool init_on_device)
        : PersistentBase<gko::array<T>, ArrayInitFunctor<T>>(
              name, db,
              ArrayInitFunctor<T>(exec, name, global_index, memory, verbose,
                                  init_on_device),
              update, verbose),
          memory_(memory),
          global_index_(global_index),
          exec_(exec),
          update_(update)
    {}

    label get_global_size() const { return global_index_.size(); }

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


    std::shared_ptr<gko::array<T>> get_array() const
    {
        return this->get_persistent_object();
    }

    /* Create a Dense<T> vector from underlying data
     * */
    std::shared_ptr<gko::matrix::Dense<T>> get_dense_vec() const
    {
        auto array = get_array();
        auto result = gko::share(vec::create(
            exec_.get_device_exec(), gko::dim<2>(array->get_num_elems(), 1),
            gko::array<T>::view(exec_.get_device_exec(), array->get_num_elems(),
                                get_data()),
            1));
        return result;
    }

    /* Copies from a distributed::Vector<T> back into raw host memory
     *
     * TODO store and use *memory
     * */
    void copy_back(const std::shared_ptr<gko::distributed::Vector<T>> dist_x)
    {
        const label nCells = global_index_.localSize();
        auto host_view = gko::array<T>::view(exec_.get_ref_exec(), nCells,
                                             const_cast<T *>(memory_));

        auto dist_device_exec = dist_x->get_executor();
        auto x_view = gko::array<T>::view(dist_device_exec, nCells,
                                          dist_x->get_local()->get_values());

        host_view = x_view;
    }

    void copy_back(const std::shared_ptr<gko::matrix::Dense<T>> device_x)
    {
        const label nCells = global_index_.localSize();
        auto host_view = gko::array<T>::view(exec_.get_ref_exec(), nCells,
                                             const_cast<T *>(memory_));

        auto x_view = gko::array<T>::view(exec_.get_device_exec(), nCells,
                                          device_x->get_values());

        host_view = x_view;
    }

    void copy_back_parallel(
        const gkoGlobalIndex &global_address, T *memory,
        const std::shared_ptr<gko::matrix::Dense<T>> device_x)
    {
        //	    std::cout << "!!!!!!!!!! copy back parallel" << std::endl;
        const label nCells = global_index_.localSize();
        auto host_view = gko::array<T>::view(exec_.get_ref_exec(), nCells,
                                             const_cast<T *>(memory));

        auto x_view = (Pstream::master())
                          ? gko::array<T>::view(exec_.get_device_exec(),
                                                global_address.size(),
                                                device_x->get_values())
                          : gko::array<T>(gko::ReferenceExecutor::create());

        if (Pstream::master()) {
            x_view.set_executor(exec_.get_ref_exec());

            // std::cout << "!!!!!!!!!! x_view before scatter back parallel" <<
            // std::endl; for (int i=0; i< x_view.get_num_elems(); i++) {
            // 	std::cout << i << ": " << host_view.get_const_data()[i] <<
            // std::endl;
            // }
        }

        global_address.scatter(x_view, host_view);

        // std::cout << "!!!!!!!!!! host_view after scatter back parallel" <<
        // std::endl; for (int i=0; i< nCells; i++) { 	std::cout << i << ": "
        // << host_view.get_const_data()[i] << std::endl;
        // }
    }

    const ExecutorHandler &get_exec_handler() const { return exec_; }
};

}  // namespace Foam

#endif
