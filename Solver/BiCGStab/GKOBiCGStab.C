// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

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
