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
    Foam::GKOIR

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOIR.C

\*---------------------------------------------------------------------------*/

#ifndef GKOIR_H
#define GKOIR_H

#include "../BaseWrapper/lduBase/GKOlduBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOIRFactory {
private:
    // executor where Ginkgo will perform the computation
    //
    const word inner_solver_;

    const scalar inner_reduction_factor_;

    const label blockSize_;

public:
    GKOIRFactory(const dictionary &dictionary)
        : inner_solver_(dictionary.lookupOrDefault("innerSolver", word("CG"))),
          inner_reduction_factor_(
              dictionary.lookupOrDefault("innerReductionFactor", scalar(1e-2))),
          blockSize_(dictionary.lookupOrDefault("maxBlockSize", label(16))){};

    std::unique_ptr<gko::solver::Ir<double>::Factory,
                    std::default_delete<gko::solver::Ir<double>::Factory>>
    create_solver(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec) const
    {
        if (inner_solver_ == "scalarJacobi") {
            return create_scalar_jacobi(exec, criterion_vec);
        }
        return create_default(exec, criterion_vec);
    };

    std::unique_ptr<gko::solver::Ir<double>::Factory,
                    std::default_delete<gko::solver::Ir<double>::Factory>>
    create_default(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec) const
    {
        return gko::solver::Ir<scalar>::build()
            .with_solver(
                gko::solver::Cg<scalar>::build()
                    .with_criteria(
                        gko::stop::ResidualNorm<scalar>::build()
                            .with_reduction_factor(inner_reduction_factor_)
                            .on(exec))
                    .on(exec))
            .with_criteria(criterion_vec)
            .on(exec);
    };
    std::unique_ptr<gko::solver::Ir<double>::Factory,
                    std::default_delete<gko::solver::Ir<double>::Factory>>
    create_scalar_jacobi(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec) const
    {
        int sweeps = 3;
        using bj = gko::preconditioner::Jacobi<>;
        auto bj_factory = bj::build()
                              .with_max_block_size(blockSize_)
                              // .with_storage_optimization(
                              //     gko::precision_reduction::autodetect())
                              .on(exec);

        return gko::solver::Ir<scalar>::build()
            .with_solver(gko::share(bj_factory))
            .with_criteria(criterion_vec)
            .on(exec);
    };
};

/*---------------------------------------------------------------------------*\
                           Class GKOIR Declaration
\*---------------------------------------------------------------------------*/


class GKOIR : public GKOlduBaseSolver<GKOIRFactory> {
    // Private Member Functions

public:
    TypeName("GKOIR");

    //- Disallow default bitwise copy construct
    GKOIR(const GKOIR &);

    //- Disallow default bitwise assignment
    void operator=(const GKOIR &);


    // Constructors

    //- Construct from matrix components and solver controls
    GKOIR(const word &fieldName, const lduMatrix &matrix,
          const FieldField<Field, scalar> &interfaceBouCoeffs,
          const FieldField<Field, scalar> &interfaceIntCoeffs,
          const lduInterfaceFieldPtrsList &interfaces,
          const dictionary &solverControls)
        : GKOlduBaseSolver(fieldName, matrix, interfaceBouCoeffs,
                           interfaceIntCoeffs, interfaces, solverControls){};

    //- Destructor
    virtual ~GKOIR(){};


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
