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

#ifndef GKOlduBase_H
#define GKOlduBase_H

#include "../StoppingCriterion/StoppingCriterion.H"
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

#define CREATE_SOLVER_METHODS(Solver)                                       \
    std::unique_ptr<Solver::Factory, std::default_delete<Solver::Factory>>  \
    create_dist_solver(                                                     \
        std::shared_ptr<gko::Executor> exec,                                \
        std::shared_ptr<gko::LinOp> gkomatrix, std::shared_ptr<dist_vec> x, \
        std::shared_ptr<dist_vec> b, const label verbose,                   \
        const bool export_res, std::shared_ptr<gko::LinOp> precond) const   \
    {                                                                       \
        stoppingCriterionVec_.push_back(                                    \
            stoppingCriterion_.build_dist_stopping_criterion(               \
                exec, gkomatrix, x, b, verbose, export_res,                 \
                get_prev_number_of_iterations(),                            \
                get_solve_prev_rel_res_cost()));                            \
                                                                            \
        if (precond != NULL) return create_precond(exec, precond);          \
        return create_default(exec);                                        \
    };


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
