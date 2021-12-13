#include "DevicePersistentArray.H"

namespace Foam {


// to store the std::shared_ptr<T> in the IO registry the type needs to be
// declared
defineTemplateTypeNameWithName(DevicePersistentBase<gko::matrix::Csr<scalar>>,
                               "PersistentCSRMatrix");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::Array<scalar>>,
                               "PersistentScalarArray");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::Array<label>>,
                               "PersistentLabelArray");
}  // namespace Foam
