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
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::GKOBiCGStab

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOBiCGStab.C

\*---------------------------------------------------------------------------*/

#ifndef GKOBiCGStab_H
#define GKOBiCGStab_H

#include "../BaseWrapper/lduBase/GKOlduBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

/*---------------------------------------------------------------------------*\
                           Class GKOBiCGStab Declaration
\*---------------------------------------------------------------------------*/
class GKOBiCGStabFactory {
private:
    using bicgstab = gko::solver::Bicgstab<scalar>;
    // executor where Ginkgo will perform the computation

    const word preconditioner_;

    const StoppingCriterion stoppingCriterion_;

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        stoppingCriterionVec_;


public:
    GKOBiCGStabFactory(const dictionary &solverControls)
        : preconditioner_(
              solverControls.lookupOrDefault("preconditioner", word("none"))),
          stoppingCriterion_(solverControls){};

    std::unique_ptr<gko::solver::Bicgstab<double>::Factory,
                    std::default_delete<gko::solver::Bicgstab<double>::Factory>>
    create_solver(std::shared_ptr<gko::Executor> exec,
                  std::shared_ptr<gko::log::Logger> logger,
                  std::shared_ptr<mtx> gkomatrix, std::shared_ptr<vec> x,
                  std::shared_ptr<vec> b, label prev_solve_iters,
                  const bool verbose, std::shared_ptr<gko::LinOp> precond) const
    {
        stoppingCriterionVec_.push_back(
            stoppingCriterion_.build_stopping_criterion(
                exec, gkomatrix, x, b, verbose, prev_solve_iters));

        if (precond != NULL) return create_precond(exec, logger, precond);
        return create_default(exec, logger);
    };

    std::unique_ptr<gko::solver::Bicgstab<double>::Factory,
                    std::default_delete<gko::solver::Bicgstab<double>::Factory>>
    create_default(std::shared_ptr<gko::Executor> exec,
                   std::shared_ptr<gko::log::Logger> logger) const
    {
        auto bicgstab = gko::solver::Bicgstab<scalar>::build()
                            .with_criteria(stoppingCriterionVec_)
                            .on(exec);
        bicgstab->add_logger(logger);
        return bicgstab;
    };

    std::unique_ptr<bicgstab::Factory, std::default_delete<bicgstab::Factory>>
    create_precond(std::shared_ptr<gko::Executor> exec,
                   std::shared_ptr<gko::log::Logger> logger,
                   std::shared_ptr<gko::LinOp> precond) const
    {
        auto bicgstab = gko::solver::Bicgstab<scalar>::build()
                            .with_criteria(stoppingCriterionVec_)
                            .with_generated_preconditioner(precond)
                            .on(exec);
        bicgstab->add_logger(logger);
        return bicgstab;
    };

    scalar get_init_res_norm() const
    {
        return stoppingCriterion_.get_init_res_norm();
    }

    scalar get_res_norm() const { return stoppingCriterion_.get_res_norm(); }
};


class GKOBiCGStab : public GKOlduBaseSolver<GKOBiCGStabFactory> {
    // Private Member Functions

public:
    TypeName("GKOBiCGStab");

    //- Disallow default bitwise copy construct
    GKOBiCGStab(const GKOBiCGStab &);

    //- Disallow default bitwise assignment
    void operator=(const GKOBiCGStab &);


    // Constructors

    //- Construct from matrix components and solver controls
    GKOBiCGStab(const word &fieldName, const lduMatrix &matrix,
                const FieldField<Field, scalar> &interfaceBouCoeffs,
                const FieldField<Field, scalar> &interfaceIntCoeffs,
                const lduInterfaceFieldPtrsList &interfaces,
                const dictionary &solverControls)
        : GKOlduBaseSolver(fieldName, matrix, interfaceBouCoeffs,
                           interfaceIntCoeffs, interfaces, solverControls){};

    //- Destructor
    virtual ~GKOBiCGStab(){};


    // Member Functions

    //- Solve the matrix with this solver

    virtual solverPerformance solve(scalarField &psi, const scalarField &source,
                                    const direction cmpt = 0) const
    {
        return solve_impl(this->typeName, psi, source, cmpt);
    }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

}  // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
