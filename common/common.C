
#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>

#include "common.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

defineTemplateTypeNameWithName(IOPtr<gko::matrix::Coo<scalar>>, "IOPtr");

// // IOPtr::addsymMatrixConstructorToTable<GKOCG>
// // addGKOCGSymMatrixConstructorToTable_;
} // namespace Foam
