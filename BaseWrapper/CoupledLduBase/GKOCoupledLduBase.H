// SPDX-License-Identifier: GPLv3 
// SPDX-FileCopyrightText: 2024 OGL authors 

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
