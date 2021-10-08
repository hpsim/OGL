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
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/

#ifndef GKOCG_H
#define GKOCG_H

#include "../BaseWrapper/lduBase/GKOlduBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOCGFactory {
private:
    using cg = gko::solver::Cg<scalar>;
    // executor where Ginkgo will perform the computation

    const word preconditioner_;

    const label blockSize_;

public:
    GKOCGFactory(const dictionary &dictionary_)
        : preconditioner_(
              dictionary_.lookupOrDefault("preconditioner", word("none"))),
          blockSize_(dictionary_.lookupOrDefault("maxBlockSize", label(16))){};

    std::unique_ptr<cg::Factory, std::default_delete<cg::Factory>>
    create_solver(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec,
        std::shared_ptr<gko::log::Logger> logger,
        std::shared_ptr<gko::LinOp> precond) const
    {
        if (precond != NULL)
            return create_precond(exec, criterion_vec, logger, precond);
        return create_default(exec, criterion_vec, logger);
    };

    std::unique_ptr<cg::Factory, std::default_delete<cg::Factory>>
    create_default(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec,
        std::shared_ptr<gko::log::Logger> logger) const
    {
        auto cg = cg::build().with_criteria(criterion_vec).on(exec);
        cg->add_logger(logger);
        return cg;
    };

    std::unique_ptr<cg::Factory, std::default_delete<cg::Factory>>
    create_precond(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec,
        std::shared_ptr<gko::log::Logger> logger,
        std::shared_ptr<gko::LinOp> precond) const
    {
        auto cg = cg::build()
                      .with_criteria(criterion_vec)
                      .with_generated_preconditioner(precond)
                      .on(exec);
        cg->add_logger(logger);
        return cg;
    };
};

/*---------------------------------------------------------------------------*\
                           Class GKOCG Declaration
\*---------------------------------------------------------------------------*/


class GKOCG : public GKOlduBaseSolver<GKOCGFactory> {
    // Private Member Functions

public:
    TypeName("GKOCG");

    //- Disallow default bitwise copy construct
    GKOCG(const GKOCG &);

    //- Disallow default bitwise assignment
    void operator=(const GKOCG &);


    // Constructors

    //- Construct from matrix components and solver controls
    GKOCG(const word &fieldName, const lduMatrix &matrix,
          const FieldField<Field, scalar> &interfaceBouCoeffs,
          const FieldField<Field, scalar> &interfaceIntCoeffs,
          const lduInterfaceFieldPtrsList &interfaces,
          const dictionary &solverControls)
        : GKOlduBaseSolver(fieldName, matrix, interfaceBouCoeffs,
                           interfaceIntCoeffs, interfaces, solverControls){};

    //- Destructor
    virtual ~GKOCG(){};


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
