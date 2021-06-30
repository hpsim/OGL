/*---------------------------------------------------------------------------*\
License
    This file is part of OGL.

    OGL is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
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

#ifndef GKOlduBase_H
#define GKOlduBase_H

#include "../IOHandler/IOExecutorHandler/IOExecutorHandler.H"
#include "../IOHandler/IOSortingIdxHandler/IOSortingIdxHandler.H"
#include "../common/StoppingCriterion.H"
#include "../common/common.H"
#include "../lduLduBase/lduLduBase.H"

#include "IOPtrList.H"
#include "fvCFD.H"
#include "lduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

/*---------------------------------------------------------------------------*\
                           Class GKOlduBaseSolver Declaration
\*---------------------------------------------------------------------------*/


template <class SolverFactory>
class GKOlduBaseSolver : public lduLduBase<lduMatrix, SolverFactory> {
public:
    //- Construct from matrix components and solver controls
    GKOlduBaseSolver(const word &fieldName, const lduMatrix &matrix,
                     const FieldField<Field, scalar> &interfaceBouCoeffs,
                     const FieldField<Field, scalar> &interfaceIntCoeffs,
                     const lduInterfaceFieldPtrsList &interfaces,
                     const dictionary &solverControls)
        : lduLduBase<lduMatrix, SolverFactory>(
              fieldName, matrix, interfaceBouCoeffs, interfaceIntCoeffs,
              interfaces, solverControls){};

    virtual solverPerformance solve_impl(word typeName, scalarField &psi,
                                         const scalarField &source,
                                         const direction cmpt = 0) const
    {
        return this->solve_impl_(typeName, psi, source, cmpt);
    };
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// //

}  // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
