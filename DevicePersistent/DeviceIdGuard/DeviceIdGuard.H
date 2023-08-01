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
    Foam::DeviceIdGuardHandler

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    DeviceIdGuardHandler.C

\*---------------------------------------------------------------------------*/

#ifndef OGL_DeviceIdGuardHandler_INCLUDED_H
#define OGL_DeviceIdGuardHandler_INCLUDED_H

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
        return std::make_shared<gko::scoped_device_id_guard>(std::move(executor_->get_scoped_device_id_guard()));
    }
};

class DeviceIdGuardHandler : public PersistentBase<gko::scoped_device_id_guard,
                             DeviceIdGuardInitFunctor > {
public:
    DeviceIdGuardHandler(const objectRegistry &db ,
                    const word field_name,
                    const std::shared_ptr<gko::Executor> executor
                    )
        : PersistentBase<gko::scoped_device_id_guard, DeviceIdGuardInitFunctor>(
              "scoped_device_id_guard_" + field_name,
              db,
              DeviceIdGuardInitFunctor(executor), true, 0)
    {}
};


}  // namespace Foam
// namespace Foam
#endif
