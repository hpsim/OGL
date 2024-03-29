#include <ginkgo/ginkgo.hpp>
#include <type_traits>

#include "ExecutorHandler.H"

namespace Foam {


// to store the std::shared_ptr<T> in the IO registry the type needs to be
// declared
defineTemplateTypeNameWithName(DevicePersistentBase<gko::Executor>,
                               "PersistentCSRMatrix");
}  // namespace Foam
