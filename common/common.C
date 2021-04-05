
#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>

#include "common.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

defineTemplateTypeNameWithName(GKOCOOIOPtr, "COOIOPtr");
defineTemplateTypeNameWithName(GKOExecPtr, "ExecIOPtr");

// // IOPtr::addsymMatrixConstructorToTable<GKOCG>
// // addGKOCGSymMatrixConstructorToTable_;
}  // namespace Foam
