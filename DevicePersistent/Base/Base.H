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
    Foam::IOExecutorHandler

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    IOExecutorHandler.C

\*---------------------------------------------------------------------------*/

#ifndef OGL_IOPtr_INCLUDED_H
#define OGL_IOPtr_INCLUDED_H

#include <ginkgo/ginkgo.hpp>
#include "fvCFD.H"

#include "common/common.H"

namespace Foam {

/* Base class for storing ginkgo and OGL data structures in the Objectregistry
 *
 * On creation it registers it self to the object registry and holds
 * a shared_ptr<T> where T is a gko data type
 *  */
template <class T>
class DevicePersistentBase : public regIOobject {
private:
    std::shared_ptr<T> ptr_;

public:
    // - Runtime type information
    TypeName("IOPtr");

    //- Construct from IOobject and a PtrList
    DevicePersistentBase(const IOobject &io, std::shared_ptr<T> in_ptr)
        : regIOobject(io), ptr_(in_ptr){};

    // get a ptr to the underlying data
    std::shared_ptr<T> get_ptr() { return ptr_; };

    bool writeData(Ostream &) const { return false; };
};

/* Base class for storing ginkgo and OGL data structures in the Objectregistry
 *
 * This class creates an unitialised Ginkgo array and copies the
 * shared ptr to the DevicePersistentBase. This will prevent it from
 * being deleted when leaving the scope.
 *
 * Additionally, for parallel runs it can be specified, whether the local,
 * global or both data should be persistent. Here are some usecase example
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

    // indicating if the underlying object was
    // restored from the object registry
    mutable bool from_registry_;

    const label verbose_;

    const InitFunctor init_functor_;

    // This needs to be a pointer so that it can be assigned
    // during the constructor, since regIOobject needs arguments
    // to be constructed
    mutable std::shared_ptr<T> persistent_object_{};


public:
    PersistentBase(const word name, const objectRegistry &db,
                   // const ExecutorHandler &exec,
                   const InitFunctor f, const bool update, const label verbose)
        : name_(name),
          db_(db),
          from_registry_{db.foundObject<regIOobject>(name_)},
          verbose_(verbose),
          init_functor_(f)
    {
        if (from_registry_) {
            word msg = "reading " + name_ + " from registry";
            MLOG_1(verbose_, msg)

            if (update) {
                persistent_object_ =
                    db_.lookupObjectRef<DevicePersistentBase<T>>(name_)
                        .get_ptr();

                TIME_WITH_FIELDNAME(verbose_, call_update, name_,
                                    f.update(persistent_object_);)
            } else {
                persistent_object_ =
                    db_.lookupObjectRef<DevicePersistentBase<T>>(name_)
                        .get_ptr();
            }

        } else {
            word msg = "initialising " + name_;
            MLOG_1(verbose_, msg)
            const fileName path = name_;

            // NOTE when objectRegistry is deleted delete for every pointer
            // owned by the registry is called
            // TODO make sure isOwnedByRegistry returns true
            TIME_WITH_FIELDNAME(verbose_, call_init, name_,
                                auto po = new DevicePersistentBase<T>(
                                    IOobject(path, db), f.init());)

            persistent_object_ = po->get_ptr();
        }
    }

    std::shared_ptr<T> get_persistent_object() const
    {
        return persistent_object_;
    }

    void set_persistent_object(std::shared_ptr<T> obj) const
    {
        persistent_object_ = obj;
    }


    bool get_verbose() const { return verbose_; }

    word get_name() const { return name_; }

    const InitFunctor &get_init_functor() const { return init_functor_; }

    const objectRegistry &get_db() const { return db_; }

    bool get_stored() const { return from_registry_; }
};

// TODO is that needed?
typedef DevicePersistentBase<gko::Executor> GKOExecPtr;
typedef DevicePersistentBase<gko::CudaExecutor> GKOCudaExecPtr;
typedef DevicePersistentBase<gko::ReferenceExecutor> GKOReferenceExecPtr;
typedef DevicePersistentBase<gko::OmpExecutor> GKOOmpExecPtr;
typedef DevicePersistentBase<gko::HipExecutor> GKOHipExecPtr;
typedef DevicePersistentBase<gko::DpcppExecutor> GKODpcppExecPtr;

}  // namespace Foam

#endif
