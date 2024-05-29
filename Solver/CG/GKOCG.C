// SPDX-License-Identifier: GPLv3 
// SPDX-FileCopyrightText: 2024 OGL authors 

#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>
#include "GKOCG.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

defineTypeNameAndDebug(GKOCG, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<GKOCG>
    addGKOCGSymMatrixConstructorToTable_;
}  // namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //


// ************************************************************************* //
