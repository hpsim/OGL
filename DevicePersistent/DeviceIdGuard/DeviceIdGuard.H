// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "DevicePersistent/Base/Base.H"
#include "fvCFD.H"

#include <ginkgo/ginkgo.hpp>


namespace Foam {

struct DeviceIdGuardInitFunctor {
    const std::shared_ptr<gko::Executor> executor_;


    DeviceIdGuardInitFunctor(const std::shared_ptr<gko::Executor> executor)
        : executor_(executor)
    {}


    void update(std::shared_ptr<gko::scoped_device_id_guard>) const {}

    std::shared_ptr<gko::scoped_device_id_guard> init() const
    {
        return std::make_shared<gko::scoped_device_id_guard>(
            std::move(executor_->get_scoped_device_id_guard()));
    }
};

class DeviceIdGuardHandler : public PersistentBase<gko::scoped_device_id_guard,
                                                   DeviceIdGuardInitFunctor> {
public:
    DeviceIdGuardHandler(const objectRegistry &db, const word field_name,
                         const std::shared_ptr<gko::Executor> executor)
        : PersistentBase<gko::scoped_device_id_guard, DeviceIdGuardInitFunctor>(
              "scoped_device_id_guard_" + field_name, db,
              DeviceIdGuardInitFunctor(executor), true, 0)
    {}
};


}  // namespace Foam
// namespace Foam