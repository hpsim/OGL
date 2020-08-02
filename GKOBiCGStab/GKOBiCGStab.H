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

#include "GKOBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

/*---------------------------------------------------------------------------*\
                           Class GKOBiCGStab Declaration
\*---------------------------------------------------------------------------*/
class GKOBiCGStabFactory {
private:
    // executor where Ginkgo will perform the computation

    const label maxIterI_;

    const scalar toleranceI_;

    const word preconditioner_;

public:
    GKOBiCGStabFactory(const dictionary &controlDict_)
        : maxIterI_(controlDict_.lookupOrDefault("maxIter", label(1000))),
          toleranceI_(controlDict_.lookupOrDefault("tolerance", scalar(1e-6))),
          preconditioner_(
              controlDict_.lookupOrDefault("preconditioner", word("none"))){};

    std::unique_ptr<gko::solver::Bicgstab<double>::Factory,
                    std::default_delete<gko::solver::Bicgstab<double>::Factory>>
    create_solver(std::shared_ptr<gko::Executor> exec) const
    {
        if (preconditioner_ == "none") return create_default(exec);
    };

    std::unique_ptr<gko::solver::Bicgstab<double>::Factory,
                    std::default_delete<gko::solver::Bicgstab<double>::Factory>>
    create_default(std::shared_ptr<gko::Executor> exec) const
    {
        return gko::solver::Bicgstab<scalar>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(maxIterI_).on(
                    exec),
                gko::stop::ResidualNormReduction<scalar>::build()
                    .with_reduction_factor(toleranceI_)
                    .on(exec))
            .on(exec);
    };

    std::unique_ptr<gko::solver::Bicgstab<double>::Factory,
                    std::default_delete<gko::solver::Bicgstab<double>::Factory>>
    create_BJ(std::shared_ptr<gko::Executor> exec) const
    {
        using bj = gko::preconditioner::Jacobi<>;
        return gko::solver::Bicgstab<scalar>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(maxIterI_).on(
                    exec),
                gko::stop::ResidualNormReduction<scalar>::build()
                    .with_reduction_factor(toleranceI_)
                    .on(exec))
            .with_preconditioner(bj::build().with_max_block_size(8u).on(exec))
            .on(exec);
    };
};



class GKOBiCGStab : public GKOBaseSolver<GKOBiCGStabFactory> {
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
        : GKOBaseSolver(fieldName, matrix, interfaceBouCoeffs,
                        interfaceIntCoeffs, interfaces, solverControls){};

    //- Destructor
    virtual ~GKOBiCGStab(){};


    // Member Functions

    //- Solve the matrix with this solver

    virtual solverPerformance solve(scalarField &psi, const scalarField &source,
                                    const direction cmpt = 0) const {
        return solve_impl(this->typeName, psi, source, cmpt);
    }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

}  // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
