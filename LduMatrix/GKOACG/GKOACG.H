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
    Foam::GKOACG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOACG.C

\*---------------------------------------------------------------------------*/

#ifndef GKOACG_H
#define GKOACG_H

#include "GKOCoupledLduBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOACGFactory {
private:
    const dictionary &solverControls_;

    const objectRegistry &db_;

    const word sysMatrixName_;

    const word preconditioner_;

public:
    GKOACGFactory(const dictionary &solverControls, const objectRegistry &db,
                  word sysMatrixName)
        : solverControls_(solverControls),
          db_(db),
          sysMatrixName_(sysMatrixName),
          preconditioner_(solverControls_.lookupOrDefault("preconditioner",
                                                          word("none"))){};

    std::unique_ptr<gko::solver::Cg<double>::Factory,
                    std::default_delete<gko::solver::Cg<double>::Factory>>
    create_solver(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec) const
    {
        if (preconditioner_ == "none")
            return create_default(exec, criterion_vec);
        if (preconditioner_ == "BJ") return create_BJ(exec, criterion_vec);
    };

    std::unique_ptr<gko::solver::Cg<double>::Factory,
                    std::default_delete<gko::solver::Cg<double>::Factory>>
    create_default(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec) const
    {
        return gko::solver::Cg<scalar>::build()
            .with_criteria(criterion_vec)
            .on(exec);
    };

    std::unique_ptr<gko::solver::Cg<double>::Factory,
                    std::default_delete<gko::solver::Cg<double>::Factory>>
    create_BJ(std::shared_ptr<gko::Executor> exec,
              std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
                  criterion_vec) const
    {
        using bj = gko::preconditioner::Jacobi<>;
        return gko::solver::Cg<scalar>::build()
            .with_criteria(criterion_vec)
            .with_preconditioner(bj::build().with_max_block_size(8u).on(exec))
            .on(exec);
    };
};

/*---------------------------------------------------------------------------*\
                           Class GKOACG Declaration
\*---------------------------------------------------------------------------*/


class GKOACG : public GKOLduBaseSolver<vector, scalar, GKOACGFactory> {
    // Private Member Functions

public:
    TypeName("GKOACG");

    //- Disallow default bitwise copy construct
    GKOACG(const GKOACG &);

    //- Disallow default bitwise assignment
    void operator=(const GKOACG &);


    // Constructors

    //- Construct from matrix components and solver controls
    GKOACG(const word &fieldName,
           const LduMatrix<vector, scalar, scalar> &matrix,
           const dictionary &solverControls)
        : GKOLduBaseSolver<vector, scalar, GKOACGFactory>(fieldName, matrix,
                                                          solverControls){};

    //- Destructor
    virtual ~GKOACG(){};


    // Member Functions

    //- Solve the matrix with this solver

    virtual SolverPerformance<vector> solve(Field<vector> &psi) const
    {
        return solve_impl(psi);
    }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

}  // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
