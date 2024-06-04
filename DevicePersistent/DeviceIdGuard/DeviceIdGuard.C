// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ginkgo/ginkgo.hpp>
#include <type_traits>

#include "DeviceIdGuard.H"

namespace Foam {


// to store the std::shared_ptr<T> in the IO registry the type needs to be
// declared
defineTemplateTypeNameWithName(
    DevicePersistentBase<gko::scoped_device_id_guard>,
    "scoped_device_id_guard");
}  // namespace Foam
