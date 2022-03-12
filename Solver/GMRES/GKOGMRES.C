/*---------------------------------------------------------------------------*\
License
    This file is part of OGL.

    OGL is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OGL is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OGL.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

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
