// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Base.H"

#include "MatrixWrapper/Distributed/Distributed.H"

namespace Foam {

// to store the std::shared_ptr<T> in the IO registry the type needs to be
// declared
defineTemplateTypeNameWithName(DevicePersistentBase<gko::matrix::Csr<scalar>>,
                               "PersistentCSRMatrix");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::array<scalar>>,
                               "PersistentScalarArray");
defineTemplateTypeNameWithName(
    DevicePersistentBase<gko::experimental::distributed::Vector<scalar>>,
    "PersistentScalarVector");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::array<label>>,
                               "PersistentLabelArray");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::matrix::Dense<label>>,
                               "PersistentLabelVec");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::matrix::Dense<scalar>>,
                               "PersistentScalarVec");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::LinOp>,
                               "PersistentLinOp");

// typedef needed  to avoid confusion with the comma separated template
// arguments as macro arguments
typedef gko::experimental::distributed::localized_partition<label> Partition;
defineTemplateTypeNameWithName(DevicePersistentBase<Partition>,
                               "PersistentPartition");

typedef RepartDistMatrix<scalar, label, label> RepartDistMatrix;
defineTemplateTypeNameWithName(DevicePersistentBase<RepartDistMatrix>,
                               "PersistentMatrix");
}  // namespace Foam
