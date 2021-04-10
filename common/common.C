
#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>

#include "common.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

void export_system(const word fieldName, const mtx *A, const vec *x,
                   const vec *b)
{
    std::string fn_mtx{fieldName + "_A.mtx"};
    std::ofstream stream{fn_mtx};
    std::cerr << "Writing " << fn_mtx << std::endl;
    gko::write(stream, A, gko::layout_type::coordinate);

    std::string fn_b{fieldName + "_b.mtx"};
    std::ofstream stream_b{fn_b};
    std::cerr << "Writing " << fn_b << std::endl;
    gko::write(stream_b, b);

    std::string fn_x{fieldName + "_x.mtx"};
    std::ofstream stream_x{fn_x};
    std::cerr << "Writing " << fn_x << std::endl;
    gko::write(stream_x, x);
};

defineTemplateTypeNameWithName(GKOCOOIOPtr, "COOIOPtr");
defineTemplateTypeNameWithName(GKOExecPtr, "ExecIOPtr");
defineTemplateTypeNameWithName(GKOCudaExecPtr, "CudaExecIOPtr");
defineTemplateTypeNameWithName(GKOOmpExecPtr, "OmpExecIOPtr");
defineTemplateTypeNameWithName(GKOHipExecPtr, "HipExecIOPtr");
defineTemplateTypeNameWithName(GKOReferenceExecPtr, "ReferenceExecIOPtr");

// // IOPtr::addsymMatrixConstructorToTable<GKOCG>
// // addGKOCGSymMatrixConstructorToTable_;
}  // namespace Foam
