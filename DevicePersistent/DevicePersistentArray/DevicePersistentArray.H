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
#include "../IOExecutorHandler/IOExecutorHandler.H"
#include "../IOGlobalIndex/gkoGlobalIndex.H"

#include "../common/common.H"

namespace Foam {
/* Base class for storing ginkgo and OGL data structures in the Objectregistry
 *
 * This class creates an unitialised Ginkgo array and copies the
 * shared ptr to the DevicePersistentBase. This will prevent it from
 * beeing deleted when leaving the scope.
 *
 * Additionally, for parallel runs it can be specified, whether the local,
 * global or both data should be persistant. Here are some usecase example
 * ldu_csr_mapping, local only
 * values, local and global persistent
 * row and col ptr only global persistent
 *
 * TODO either make storing optional or the gather method static
 * */
template <class T, class InitFunctor>
class PersistentBase {
private:
    mutable word name_;

    const objectRegistry &db_;

    // executor where the array is initialised on
    const IOExecutorHandler &exec_;

    // indicating if the underlying object was
    // restored from the object registry
    mutable bool from_registry_;

    const label verbose_;

    // This needs to be a pointer so that it can be assigned
    // during the constructor, since regIOobject needs arguments
    // to be constructed
    mutable std::shared_ptr<T> persistent_object_{};


public:
    PersistentBase(const word name, const objectRegistry &db,
                   const IOExecutorHandler &exec, const InitFunctor f,
                   const bool update, const label verbose)
        : name_(name),
          db_(db),
          exec_(exec),
          from_registry_{db.foundObject<regIOobject>(name_)},
          verbose_(verbose)
    {
        if (from_registry_) {
            word msg = "reading " + name_ + " from registry";
            LOG_1(verbose_, msg)

            if (update) {
                persistent_object_ =
                    db_.lookupObjectRef<DevicePersistentBase<T>>(name_)
                        .get_ptr();
                f.update(persistent_object_);
            } else {
                persistent_object_ =
                    db_.lookupObjectRef<DevicePersistentBase<T>>(name_)
                        .get_ptr();
            }

        } else {
            word msg = "initialising " + name_;
            LOG_1(verbose_, msg)
            const fileName path = name_;

            persistent_object_ =
                DevicePersistentBase<T>(IOobject(path, db), f.init()).get_ptr();
        }
    }

    std::shared_ptr<gko::Executor> get_device_exec() const
    {
        return exec_.get_device_exec();
    }

    std::shared_ptr<gko::Executor> get_ref_exec() const
    {
        return exec_.ref_exec();
    }


    std::shared_ptr<T> get_persistent_object() const
    {
        return persistent_object_;
    }

    bool get_verbose() const { return verbose_; }

    word get_name() const { return name_; }

    const objectRegistry &get_db() const { return db_; }

    bool get_stored() const { return from_registry_; }

    // copy_values back to
};

template <class T>
struct ArrayInitFunctor {
    const IOExecutorHandler &exec_;

    const gkoGlobalIndex &global_index_;

    const label verbose_;

    const bool on_device_;

    // Memory
    const T *other_;


    ArrayInitFunctor(const IOExecutorHandler &exec,
                     const gkoGlobalIndex &global_index, const label verbose,
                     const bool on_device = false)
        : exec_(exec),
          global_index_(global_index),
          on_device_(on_device),
          verbose_(verbose),
          other_(NULL)
    {}

    ArrayInitFunctor(const IOExecutorHandler &exec,
                     const gkoGlobalIndex &global_index, const T *other,
                     const label verbose, const bool on_device = false)
        : exec_(exec),
          global_index_(global_index),
          on_device_(on_device),
          verbose_(verbose),
          other_(other)
    {}


    // update persistent array from host memory
    void update(std::shared_ptr<gko::Array<T>> persistent_array) const
    {
        if (other_ != NULL) {
            word msg{"updating array of size " +
                     std::to_string(global_index_.localSize())};
            LOG_1(verbose_, msg)

            auto host_view =
                gko::Array<T>::view(exec_.ref_exec(), global_index_.localSize(),
                                    const_cast<T *>(other_));
            persistent_array->operator=(host_view);
        }
    }

    std::shared_ptr<gko::Array<T>> init() const
    {
        auto exec = (on_device_) ? exec_.get_device_exec() : exec_.ref_exec();
        word msg{"initialising on array of size " +
                 std::to_string(global_index_.localSize())};
        msg += (on_device_) ? " on device" : " on host";
        LOG_1(verbose_, msg)

        if (other_ != NULL) {
            auto host_view =
                gko::Array<T>::view(exec_.ref_exec(), global_index_.localSize(),
                                    const_cast<T *>(other_));
            return std::make_shared<gko::Array<T>>(exec, host_view);
        } else {
            return std::make_shared<gko::Array<T>>(exec,
                                                   global_index_.localSize());
        }
    }
};

template <class T>
class PersistentArray
    : public PersistentBase<gko::Array<T>, ArrayInitFunctor<T>> {
    const gkoGlobalIndex &global_index_;

    // indicating if the underlying array needs to
    // updated even if was found in the object registry
    const bool update_;


    // TODO store_global_on_device
    // to store global on the device the data has to be inialized
    // first locally, then gathered, and finally moved to the device
    //
    const bool store_global_;


    // The same as dpb but for the global data
    // NOTE if on the host:
    //  - This should exist with the full length only on the master
    // on the device:
    //  - global data is needed on the device for example for the
    //    col and row idxs
    mutable std::shared_ptr<gko::Array<T>> persistent_object_global_{};


public:
    PersistentArray(const word name, const objectRegistry &db,
                    const IOExecutorHandler &exec,
                    const gkoGlobalIndex &global_index, const label verbose,
                    const bool update = false, const bool global = false)
        : PersistentBase<gko::Array<T>, ArrayInitFunctor<T>>(
              name, db, exec, ArrayInitFunctor<T>(exec, global_index, verbose),
              update, verbose),
          global_index_(global_index),
          update_(update),
          store_global_(false)
    {}

    PersistentArray(const word name, const objectRegistry &db,
                    const IOExecutorHandler &exec,
                    const gkoGlobalIndex &global_index, const T *memory,
                    const label verbose, const bool update = false,
                    const bool on_device = false)
        : PersistentBase<gko::Array<T>, ArrayInitFunctor<T>>(
              name, db, exec,
              ArrayInitFunctor<T>(exec, global_index, memory, verbose,
                                  on_device),
              update, verbose),
          global_index_(global_index),
          update_(update),
          store_global_(false)
    {}

    // gather the underlying data and return a
    // shared_ptr to the corresponding global array
    // currently the shared_ptr is a nullptr on all but the main rank
    std::shared_ptr<gko::Array<T>> get_global_array() const
    {
        if (Pstream::parRun()) {
            if (Pstream::master()) {
                const word global_name = this->get_name() + "_global";
                auto &db = this->get_db();
                bool stored{db.template foundObject<regIOobject>(global_name)};

                if (stored) {
                    LOG_1(
                        this->get_verbose(),
                        "global array stored, retrieving from object registry")
                    persistent_object_global_ =
                        db.template lookupObjectRef<
                              DevicePersistentBase<gko::Array<T>>>(global_name)
                            .get_ptr();
                } else {
                    LOG_1(this->get_verbose(),
                          "initialising global array " + global_name +
                              " of size " +
                              std::to_string(global_index_.size()))
                    // create a new unitialised array
                    persistent_object_global_ = std::make_shared<gko::Array<T>>(
                        this->get_ref_exec(), global_index_.size());

                    const fileName path = global_name;
                    if (store_global_) {
                        DevicePersistentBase<gko::Array<T>>(
                            IOobject(path, db), persistent_object_global_);
                    }
                }
            }

            LOG_1(this->get_verbose(), "gather global array")
            global_index_.gather<T>(this->get_persistent_object(),
                                    persistent_object_global_);
            LOG_1(this->get_verbose(), "gather global array done")

            return persistent_object_global_;
        }
        return this->get_persistent_object();
    };

    label get_global_size() const { return global_index_.size(); }

    bool get_update() const { return update_; }

    T *get_data() const { return this->get_persistent_object()->get_data(); }

    std::shared_ptr<gko::Array<T>> get_array() const
    {
        return this->get_persistent_object();
    }
    std::shared_ptr<gko::matrix::Dense<T>> get_global_dense_vec() const
    {
        auto host_global_array = get_global_array();

        LOG_1(this->get_verbose(), "create global vec")
        if (Pstream::master()) {
            return gko::share(
                vec::create(this->get_device_exec(),
                            gko::dim<2>(host_global_array->get_num_elems(), 1),
                            *host_global_array.get(), 1));
        } else {
            return {};
        }
    }

    std::shared_ptr<gko::matrix::Dense<T>> get_dense_vec() const
    {
        auto host_array = get_array();
        return gko::share(vec::create(
            this->get_device_exec(),
            gko::dim<2>(host_array->get_num_elems(), 1), *host_array.get(), 1));
    }

    const T *get_const_data() const
    {
        return this->get_persistent_object()->get_const_data();
    };

    void copy_back(T *memory,
                   const std::shared_ptr<gko::matrix::Dense<T>> device_x)
    {
        const label nCells = global_index_.localSize();
        auto host_view = gko::Array<T>::view(this->get_ref_exec(), nCells,
                                             const_cast<T *>(memory));

        auto x_view = gko::Array<T>::view(this->get_ref_exec(), nCells,
                                          device_x->get_values());

        host_view = x_view;
    }
};

}  // namespace Foam

#endif
