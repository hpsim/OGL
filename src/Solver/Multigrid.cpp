// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ginkgo/ginkgo.hpp>

#include "OGL/Solver/Multigrid.hpp"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

defineTypeNameAndDebug(GKOMultigrid, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<GKOMultigrid>
    addGKOMultigridSymMatrixConstructorToTable_;
}  // namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //


// ************************************************************************* //
