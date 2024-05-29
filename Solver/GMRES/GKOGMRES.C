// SPDX-License-Identifier: GPLv3 
// SPDX-FileCopyrightText: 2024 OGL authors 

#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>
#include "GKOGMRES.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

defineTypeNameAndDebug(GKOGMRES, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<GKOGMRES>
    addGKOGMRESSymMatrixConstructorToTable_;

lduMatrix::solver::addasymMatrixConstructorToTable<GKOGMRES>
    addGKOGMRESAsymMatrixConstructorToTable_;
}  // namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //


// ************************************************************************* //
