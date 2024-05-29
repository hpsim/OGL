// SPDX-License-Identifier: GPLv3 
// SPDX-FileCopyrightText: 2024 OGL authors 

#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>
#include "GKOACG.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

defineTypeNameAndDebug(GKOACG, 0);

LduMatrix<vector, scalar,
          scalar>::solver::addsymMatrixConstructorToTable<GKOACG>
    addGKOACGSymMatrixConstructorToTable_;
}  // namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //


// ************************************************************************* //
