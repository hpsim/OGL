// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "OGL/DevicePersistent/ExecutorHandler.H"
#include "OGL/MatrixWrapper/Distributed.H"
#include "OGL/MatrixWrapper/HostMatrix.H"

#include "fvCFD.H"

namespace Foam {

struct DistributedMatrixInitFunctor {
    const objectRegistry &db_;

    const ExecutorHandler &exec_;

    std::shared_ptr<const Repartitioner> repartitioner_;

    const word field_name_;

    const label verbose_;

    const dictionary &solver_controls_;

    const std::shared_ptr<HostMatrixWrapper> host_A_;

    DistributedMatrixInitFunctor(
        const objectRegistry &db, const ExecutorHandler &exec,
        std::shared_ptr<const Repartitioner> repartitioner, const label verbose,
        const word field_name, const std::shared_ptr<HostMatrixWrapper> host_A,
        const dictionary &solver_controls)
        : db_(db),
          exec_(exec),
          repartitioner_(repartitioner),
          field_name_(field_name),
          verbose_(verbose),
          solver_controls_(solver_controls),
          host_A_(host_A)
    {}

    void update(
        std::shared_ptr<RepartDistMatrix> persistent_device_matrix) const
    {
        persistent_device_matrix->update(exec_, repartitioner_, host_A_);
        // persistent_device_matrix->operator=(
        //  *RepartDistMatrix<scalar, label, label>::create(
        //     solver_controls_.lookupOrDefault<word>("matrixFormat", "Coo"),
        //     repartitioner_, host_A_.get()).get());
    }

    std::shared_ptr<RepartDistMatrix> init() const
    {
        return create_distributed(
            exec_, repartitioner_, host_A_,
            solver_controls_.lookupOrDefault<word>("matrixFormat", "Coo"));
    }
};


/* This Class makes the Host LDUMatrix device persistent. It basically only
 * provides a functor to call HostMatrixWrapper::create and LduMatrix::update
 */
class PersistentDistributedMatrix {
private:
    const objectRegistry &db_;

    const label verbose_;

    const word field_name_;

    mutable PersistentBase<RepartDistMatrix, DistributedMatrixInitFunctor>
        host_matrix_;


public:
    PersistentDistributedMatrix(
        const objectRegistry &db, const ExecutorHandler &exec,
        std::shared_ptr<const Repartitioner> repartitioner,
        const word field_name, std::shared_ptr<HostMatrixWrapper> host_A,
        const dictionary &controlDict, const label verbose)
        : db_(db),
          verbose_(verbose),
          field_name_(field_name),
          host_matrix_{
              field_name + "_distributed_matrix", db,
              DistributedMatrixInitFunctor(db, exec, repartitioner, verbose_,
                                           field_name + "_distributed_matrix",
                                           host_A, controlDict),
              controlDict.lookupOrDefault<Switch>("updateSysMatrix", true),
              verbose_}
    {}

    std::shared_ptr<RepartDistMatrix> get()
    {
        return host_matrix_.get_persistent_object();
    }
};

}  // namespace Foam
