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
    using bj = gko::preconditioner::Jacobi<>;

private:
    // executor where Ginkgo will perform the computation

    const word preconditioner_;

    const label blockSize_;

public:
    GKOBiCGStabFactory(const dictionary &dictionary_)
        : preconditioner_(
              dictionary_.lookupOrDefault("preconditioner", word("none"))),
          blockSize_(dictionary_.lookupOrDefault("maxBlockSize", label(16))){};

    std::unique_ptr<gko::solver::Bicgstab<double>::Factory,
                    std::default_delete<gko::solver::Bicgstab<double>::Factory>>
    create_solver(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec,
        std::shared_ptr<const bj> precond) const
    {
        if (precond != NULL)
            return create_precond(exec, criterion_vec, precond);
        return create_default(exec, criterion_vec);
    };

    std::unique_ptr<gko::solver::Bicgstab<double>::Factory,
                    std::default_delete<gko::solver::Bicgstab<double>::Factory>>
    create_default(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec) const
    {
        return gko::solver::Bicgstab<scalar>::build()
            .with_criteria(criterion_vec)
            .on(exec);
    };

    std::unique_ptr<gko::solver::Bicgstab<double>::Factory,
                    std::default_delete<gko::solver::Bicgstab<double>::Factory>>
    create_precond(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec,
        std::shared_ptr<const bj> precond) const
    {
        return gko::solver::Bicgstab<scalar>::build()
            .with_criteria(criterion_vec)
            .with_generated_preconditioner(precond)
            .on(exec);
    };
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
