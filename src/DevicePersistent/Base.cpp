// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/DevicePersistent/Base.hpp"

namespace Foam {

// to store the std::shared_ptr<T> in the IO registry the type needs to be
// declared
defineTemplateTypeNameWithName(
    DevicePersistentBase<gko::experimental::distributed::Vector<scalar>>,
    "PersistentScalarVector");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::LinOp>,
                               "PersistentLinOp");

// typedef needed  to avoid confusion with the comma separated template
// arguments as macro arguments
typedef gko::experimental::distributed::Matrix<scalar, label, label> GkoMatrix;
defineTemplateTypeNameWithName(DevicePersistentBase<GkoMatrix>,
                               "PersistentMatrix");
}  // namespace Foam
