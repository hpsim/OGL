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
    IOPtr(const IOobject &io, std::shared_ptr<T> in_ptr)
        : regIOobject(io), ptr_(in_ptr){};

    // get a ptr to the underlying data
    std::shared_ptr<T> get_ptr() { return ptr_; };

    bool writeData(Ostream &) const { return false; };
};


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
template <class T>
class PersistentArrayBase {
private:
    mutable word name_;

    const objectRegistry &db_;

    // executor where the array is initialised on
    const std::shared_ptr<gko::Executor> device_exec_;

    const std::shared_ptr<gko::Executor> host_exec_;

    const gkoGlobalIndex &global_index_;

    // indicating if the underlying array needs to
    // updated even if was found in the object registry
    const bool update_;

    // indicating if the underlying array was
    // restored from the object registry
    mutable bool from_registry_;

    const label storage_mode;

    // This needs to be a pointer so that it can be assigned
    // during the constructor, since regIOobject needs arguments
    // to be constructed
    mutable std::shared_ptr<gko::Array<T>> dpb_{};

    // The same as dpb but for the global data
    // NOTE if on the host:
    //  - This should exist with the full length only on the master
    // on the device:
    //  - global data is needed on the device for example for the
    //    col and row idxs
    mutable std::shared_ptr<gko::Array<T>> dpb_global_{};

    // TODO store_global_on_device
    // to store global on the device the data has to be inialized
    // first locally, then gathered, and finally moved to the device
    //
    const bool store_global_on_device;


public:
    PersistentArray(const word name, const objectRegistry &db,
                    const std::shared_ptr<gko::Executor> host_exec_,
                    const gkoGlobalIndex &global_index,
                    const bool update = false, const bool global = false)
        : name_(name),
          db_(db),
          host_exec_(host_exec),
          global_index_(global_index),
          update_(update),
          from_registry_{db.foundObject<regIOobject>(name_)}
    {

        if (from_registry_) {
            dpb_ =
                db_.lookupObjectRef<DevicePersistenBase<gko::Array<T>>>(name_);
        } else {
            const fileName path = name_;

            auto array =
                std::make_shared<gko::Array<T>>(global_index_.localSize());

            dpb_ = DevicePersistenBase<gko::Array<T>>(IOobject(path, db), array)
                       .get_ptr();
        }
    }

    // gather the underlying data and return a
    // shared_ptr to the corresponding global array
    // currently the shared_ptr is a nullptr on all but the main rank
    std::shared_ptr<gko::Array<T>> get_global()
    {
        if (Pstream::parRun()) {
            if (Pstream::master()) {
                const word global_name = name + "_global";
                bool stored{db_.foundObject<regIOobject>(global_name)};

                if (stored) {
                    dpb_global_ =
                        db.lookupObjectRef<DevicePersistenBase<gko::Array<T>>>(
                              global_name)
                            .get_ptr();
                } else {
                    // create a new unitialised array
                    dpb_global_ =
                        std::make_shared<gko::Array<T>>(global_index_.size());
                    if (store_global_) {
                        DevicePersistenBase<gko::Array<T>>(IOobject(path, db),
                                                           dpb_global);
                    }
                }
            }

            global_index_.gather(dpb_, dpb_global_);

            return dpd_global_;
        }
        return dpd_;
    };

    bool get_stored() {return from_registry_;}

    bool get_update() {return update_;}

    T *get_data() const { return dpb_->get_data(); };

    const T *get_const_data() const
    {
        return dpb_.get_ptr()->get_const_data();
    };

}

typedef IOPtr<gko::Executor>
    GKOExecPtr;
typedef IOPtr<gko::CudaExecutor> GKOCudaExecPtr;
typedef IOPtr<gko::ReferenceExecutor> GKOReferenceExecPtr;
typedef IOPtr<gko::OmpExecutor> GKOOmpExecPtr;
typedef IOPtr<gko::HipExecutor> GKOHipExecPtr;
typedef IOPtr<gko::DpcppExecutor> GKODpcppExecPtr;
typedef IOPtr<idx_array> GKOIDXIOPtr;
typedef IOPtr<gko::matrix::Csr<scalar>> GKOCSRIOPtr;
typedef IOPtr<gko::matrix::Dense<scalar>> GKOVECIOPtr;

typedef IOPtr<gko::preconditioner::Jacobi<>> GKOBJIOPtr;
typedef IOPtr<gko::preconditioner::Ilu<>> GKOILUIOPtr;
typedef IOPtr<gko::LinOp> GKOLinOPIOPtr;


}  // namespace Foam

#endif
