// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

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
