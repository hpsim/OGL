// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/Solver/GMRES.hpp"

#include <ginkgo/ginkgo.hpp>

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
