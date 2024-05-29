// SPDX-License-Identifier: GPLv3 
// SPDX-FileCopyrightText: 2024 OGL authors 

#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>
#include "GKOIR.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

defineTypeNameAndDebug(GKOIR, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<GKOIR>
    addGKOIRSymMatrixConstructorToTable_;
}  // namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //


// ************************************************************************* //
