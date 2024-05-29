// SPDX-License-Identifier: GPLv3 
// SPDX-FileCopyrightText: 2024 OGL authors 

#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>
#include "GKOMultigrid.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

defineTypeNameAndDebug(GKOMultigrid, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<GKOMultigrid>
    addGKOMultigridSymMatrixConstructorToTable_;
}  // namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //


// ************************************************************************* //
