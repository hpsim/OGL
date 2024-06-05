// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef GKOlduBase_H
#define GKOlduBase_H

#include "StoppingCriterion/StoppingCriterion.H"
#include "common/common.H"
#include "lduLduBase/lduLduBase.H"

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
