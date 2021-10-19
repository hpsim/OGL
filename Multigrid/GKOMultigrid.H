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
#include "../common/StoppingCriterion.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOMultigridFactory {
private:
    using fcg = gko::solver::Fcg<scalar>;
    using cg = gko::solver::Cg<scalar>;
    using ir = gko::solver::Ir<scalar>;
    using mg = gko::solver::Multigrid;
    using bj = gko::preconditioner::Jacobi<scalar, label>;
    using amgx_pgm = gko::multigrid::AmgxPgm<scalar, label>;

    const dictionary &solverControls_;

    const dictionary &innerSolverControls_;

    const objectRegistry &db_;

    const word sysMatrixName_;

    const StoppingCriterion outerStoppingCriterion_;

    const StoppingCriterion innerStoppingCriterion_;

    const word smoother_solver_;

    const label max_block_size_;

    const scalar inner_relaxation_factor_;

    const scalar smoother_relaxation_factor_;

    const label smoother_max_iters_;

    const label coarse_max_iters_;

    const label max_levels_;

    const label min_coarse_rows_;

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        outerStoppingCriterionVec_ = {};

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        innerStoppingCriterionVec_ = {};

public:
    GKOMultigridFactory(const dictionary &solverControls,
                        const objectRegistry &db, word sysMatrixName)
        : solverControls_(solverControls),
          innerSolverControls_(solverControls.subDict("inner")),
          db_(db),
          sysMatrixName_(sysMatrixName),
          outerStoppingCriterion_(solverControls),
          innerStoppingCriterion_(StoppingCriterion(innerSolverControls_)),
          smoother_solver_(innerSolverControls_.lookupOrDefault(
              "smootherSolver", word("CG"))),
          max_block_size_(
              innerSolverControls_.lookupOrDefault("maxBlockSize", label(4))),
          inner_relaxation_factor_(innerSolverControls_.lookupOrDefault(
              "innerRelaxationFactor", scalar(0.9))),
          smoother_relaxation_factor_(innerSolverControls_.lookupOrDefault(
              "smootherRelaxationFactor", scalar(0.9))),
          smoother_max_iters_(innerSolverControls_.lookupOrDefault(
              "smootherMaxIters", label(2))),
          coarse_max_iters_(
              innerSolverControls_.lookupOrDefault("coarseMaxIters", label(4))),
          max_levels_(
              innerSolverControls_.lookupOrDefault("maxLevels", label(9))),
          min_coarse_rows_(innerSolverControls_.lookupOrDefault("minCoarseRows",
                                                                label(10))){};

    // Create solver
    std::unique_ptr<mg::Factory, std::default_delete<mg::Factory>>
    create_solver(std::shared_ptr<gko::Executor> exec,
                  std::shared_ptr<mtx> gkomatrix, std::shared_ptr<vec> x,
                  std::shared_ptr<vec> b, const bool verbose,
                  const bool export_res,
                  std::shared_ptr<gko::LinOp> precond) const
    {
        outerStoppingCriterionVec_.push_back(
            outerStoppingCriterion_.build_stopping_criterion(
                exec, gkomatrix, x, b, verbose, export_res, 0));

        auto inner_solver_gen =
            gko::share(bj::build()
                           .with_skip_sorting(true)
                           .with_max_block_size(max_block_size_)
                           .on(exec));

        auto smoother_gen = gko::share(
            ir::build()
                .with_solver(inner_solver_gen)
                .with_relaxation_factor(smoother_relaxation_factor_)
                .with_criteria(gko::stop::Iteration::build()
                                   .with_max_iters(smoother_max_iters_)
                                   .on(exec))
                .on(exec));
        // Create MultigridLevel factory
        auto mg_level_gen = amgx_pgm::build()
                                .with_deterministic(true)
                                .with_skip_sorting(true)
                                .on(exec);
        // Create CoarsestSolver factory
        auto coarsest_gen =
            gko::share(ir::build()
                           .with_solver(inner_solver_gen)
                           .with_relaxation_factor(inner_relaxation_factor_)
                           .with_criteria(gko::stop::Iteration::build()
                                              .with_max_iters(coarse_max_iters_)
                                              .on(exec))
                           .on(exec));

        // Create multigrid factory
        auto ret = mg::build()
                       .with_max_levels(max_levels_)
                       .with_min_coarse_rows(min_coarse_rows_)
                       .with_pre_smoother(smoother_gen)
                       .with_post_uses_pre(true)
                       .with_mg_level(gko::share(mg_level_gen))
                       .with_coarsest_solver(coarsest_gen)
                       .with_zero_guess(false)
                       .with_criteria(outerStoppingCriterionVec_)
                       .on(exec);

        return ret;
    };

    scalar get_init_res_norm() const
    {
        return outerStoppingCriterion_.get_init_res_norm();
    }

    scalar get_res_norm() const
    {
        return outerStoppingCriterion_.get_res_norm();
    }

    std::shared_ptr<vec> get_res_norms() const
    {
        return outerStoppingCriterion_.get_res_norms();
    }

    void store_number_of_iterations() const
    {
        set_solve_prev_iters(sysMatrixName_, db_,
                             innerStoppingCriterion_.get_num_iters());
    }

    label get_number_of_iterations() const
    {
        return outerStoppingCriterion_.get_num_iters();
    }
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
