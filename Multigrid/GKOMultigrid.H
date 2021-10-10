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
    Foam::GKOMultigrid

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOMultigrid.C

\*---------------------------------------------------------------------------*/

#ifndef OGL_IOGKOMultigrid_INCLUDED_H
#define OGL_IOGKOMultigrid_INCLUDED_H

#include "../BaseWrapper/lduBase/GKOlduBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOMultigridFactory {
private:
    // executor where Ginkgo will perform the computation
    //
    const word smoother_solver_;

    const scalar smoother_relaxation_factor_;

    const label smoother_max_iters_;

    const label max_levels_;

    const label min_coarse_rows_;

public:
    GKOMultigridFactory(const dictionary &dictionary)
        : smoother_solver_(
              dictionary.lookupOrDefault("smootherSolver", word("CG"))),
          smoother_relaxation_factor_(dictionary.lookupOrDefault(
              "smootherRelaxationFactor", scalar(0.9))),
          smoother_max_iters_(
              dictionary.lookupOrDefault("smootherMaxIters", label(2))),
          max_levels_(dictionary.lookupOrDefault("maxLevels", label(9))),
          min_coarse_rows_(
              dictionary.lookupOrDefault("minCoarseRows", label(10))){};

    // Create solver
    std::unique_ptr<gko::solver::Ir<double>::Factory,
                    std::default_delete<gko::solver::Ir<double>::Factory>>
    create_solver(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec,
        std::shared_ptr<gko::LinOp> precond) const
    {
        auto inner_solver_gen =
            gko::share(bj::build().with_max_block_size(1u).on(exec));
        auto smoother_gen = gko::share(
            ir::build()
                .with_solver(inner_solver_gen)
                .with_relaxation_factor(smoother_relaxation_factor_)
                .with_criteria(gko::stop::Iteration::build()
                                   .with_max_iters(smoother_max_iters_)
                                   .on(exec))
                .on(exec));
        // Create MultigridLevel factory
        auto mg_level_gen = amgx_pgm::build().with_deterministic(true).on(exec);
        // Create CoarsestSolver factory
        auto coarsest_gen = gko::share(
            ir::build()
                .with_solver(inner_solver_gen)
                .with_relaxation_factor(inner_relaxation_factor_)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(4u).on(exec))
                .on(exec));

        // Create multigrid factory
        return mg::build()
            .with_max_levels(max_levels_)
            .with_min_coarse_rows(min_coarse_rows_)
            .with_pre_smoother(smoother_gen)
            .with_post_uses_pre(true)
            .with_mg_level(gko::share(mg_level_gen))
            .with_coarsest_solver(coarsest_gen)
            .with_zero_guess(true)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(exec));
    };
};

/*---------------------------------------------------------------------------*\
                           Class GKOMultigrid Declaration
\*---------------------------------------------------------------------------*/


class GKOMultigrid : public GKOlduBaseSolver<GKOMultigridFactory> {
    // Private Member Functions

public:
    TypeName("GKOMultigrid");

    //- Disallow default bitwise copy construct
    GKOMultigrid(const GKOMultigrid &);

    //- Disallow default bitwise assignment
    void operator=(const GKOMultigrid &);


    // Constructors

    //- Construct from matrix components and solver controls
    GKOMultigrid(const word &fieldName, const lduMatrix &matrix,
                 const FieldField<Field, scalar> &interfaceBouCoeffs,
                 const FieldField<Field, scalar> &interfaceIntCoeffs,
                 const lduInterfaceFieldPtrsList &interfaces,
                 const dictionary &solverControls)
        : GKOlduBaseSolver(fieldName, matrix, interfaceBouCoeffs,
                           interfaceIntCoeffs, interfaces, solverControls){};

    //- Destructor
    virtual ~GKOMultigrid(){};


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
