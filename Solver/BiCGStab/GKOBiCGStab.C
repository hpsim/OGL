// SPDX-License-Identifier: GPLv3 
// SPDX-FileCopyrightText: 2024 OGL authors 

#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>
#include "GKOBiCGStab.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

defineTypeNameAndDebug(GKOBiCGStab, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<GKOBiCGStab>
    addGKOBiCGStabSymMatrixConstructorToTable_;

lduMatrix::solver::addasymMatrixConstructorToTable<GKOBiCGStab>
    addGKOBiCGStabAsymMatrixConstructorToTable_;
}  // namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //


// ************************************************************************* //
