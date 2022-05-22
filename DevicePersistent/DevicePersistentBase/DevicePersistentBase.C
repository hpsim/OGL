
#include "DevicePersistentBase.H"

namespace Foam {


// to store the std::shared_ptr<T> in the IO registry the type needs to be
// declared
defineTemplateTypeNameWithName(DevicePersistentBase<gko::matrix::Csr<scalar>>,
                               "PersistentCSRMatrix");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::Array<scalar>>,
                               "PersistentScalarArray");
defineTemplateTypeNameWithName(
    DevicePersistentBase<gko::distributed::Vector<scalar>>,
    "PersistentScalarVector");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::Array<label>>,
                               "PersistentLabelArray");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::matrix::Dense<label>>,
                               "PersistentLabelVec");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::matrix::Dense<scalar>>,
                               "PersistentScalarVec");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::LinOp>,
                               "PersistentLinOp");
}  // namespace Foam
