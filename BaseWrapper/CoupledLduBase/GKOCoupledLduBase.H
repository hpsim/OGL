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

Class
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/

#ifndef GKO_LDUBase_H
#define GKO_LDUBase_H

#include "common/common.H"
#include "lduLduBase/lduLduBase.H"

#include "LduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

/*---------------------------------------------------------------------------*\
                           Class GKOCG Declaration
\*---------------------------------------------------------------------------*/

template <class Type, class LDUType, class SolverFactory>
class GKOLduBaseSolver
    : public lduLduBase<LduMatrix<Type, LDUType, LDUType>, SolverFactory> {
public:
    GKOLduBaseSolver(const word &fieldName,
                     const LduMatrix<Type, LDUType, LDUType> &matrix,
                     const dictionary &solverControls)
        : lduLduBase<LduMatrix<Type, LDUType, LDUType>, SolverFactory>(
              fieldName, matrix, solverControls)
    {}

    virtual SolverPerformance<Type> solve_impl(Field<Type> &psi) const
    {
        return this->solve_impl_(psi);
    }
};

}  // End namespace Foam

#endif
