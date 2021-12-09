#include "DevicePersistentArray.H"

namespace Foam {


// to store the std::shared_ptr<T> in the IO registry the type needs to be
// declared
defineTemplateTypeNameWithName(DevicePersistentBase<gko::matrix::Csr<scalar>>,
                               "PersistenCSRMatrix");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::Array<scalar>>,
                               "PersistenScalarArray");
defineTemplateTypeNameWithName(DevicePersistentBase<gko::Array<label>>,
                               "PersistenLabelArray");
}  // namespace Foam
