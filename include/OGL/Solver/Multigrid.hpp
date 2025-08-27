// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "OGL/GKOlduBase.hpp"
#include "OGL/MatrixWrapper/Distributed.hpp"
#include "OGL/StoppingCriterion.hpp"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOMultigridFactory {
private:
    using cg = gko::solver::Cg<scalar>;
    using mtx = gko::matrix::Csr<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    using ir = gko::solver::Ir<scalar>;
    using mg = gko::solver::Multigrid;
    using bj = gko::preconditioner::Jacobi<scalar, label>;
    using pgm = gko::multigrid::Pgm<scalar, label>;
    using ras =
        gko::experimental::distributed::preconditioner::Schwarz<scalar, label,
                                                                label>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;

    const dictionary &solverControls_;

    const dictionary &innerSolverControls_;

    const objectRegistry &db_;

    const word sysMatrixName_;

    const StoppingCriterion outerStoppingCriterion_;

    const StoppingCriterion innerStoppingCriterion_;

    const label verbose_;

    const word coarsest_solver_;

    const word smoother_solver_;

    const label max_block_size_;

    const scalar inner_reduction_factor_;

    const scalar inner_relaxation_factor_;

    const scalar smoother_relaxation_factor_;

    const label smoother_max_iters_;

    const label coarse_max_iters_;

    const label max_levels_;

    const label min_coarse_rows_;

    const word cycle_;

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
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          coarsest_solver_(
              solverControls_.lookupOrDefault("coarsestSolver", word("CG"))),
          smoother_solver_(innerSolverControls_.lookupOrDefault(
              "smootherSolver", word("CG"))),
          max_block_size_(
              innerSolverControls_.lookupOrDefault("maxBlockSize", label(4))),
          inner_reduction_factor_(innerSolverControls_.lookupOrDefault(
              "reductionFactor", scalar(0.0001))),
          inner_relaxation_factor_(innerSolverControls_.lookupOrDefault(
              "innerRelaxationFactor", scalar(0.9))),
          smoother_relaxation_factor_(innerSolverControls_.lookupOrDefault(
              "smootherRelaxationFactor", scalar(0.9))),
          smoother_max_iters_(innerSolverControls_.lookupOrDefault(
              "smootherMaxIters", label(2))),
          coarse_max_iters_(innerSolverControls_.lookupOrDefault(
              "coarseMaxIters", label(50))),
          max_levels_(
              innerSolverControls_.lookupOrDefault("maxLevels", label(9))),
          min_coarse_rows_(
              innerSolverControls_.lookupOrDefault("minCoarseRows", label(10))),
          cycle_(solverControls_.lookupOrDefault("cycle", word("v")))
    {
        auto mtx_format =
            solverControls.lookupOrDefault("matrixFormat", word("Coo"));
        if (mtx_format != "Csr") {
            FatalErrorInFunction
                << "Ginkgos Multigrid solver currently only supports Csr "
                   "matrices make sure to set: 'matrixFormat Csr;'"
                << abort(FatalError);
        }

        word msg = std::string("Multigrid parameters:") +
                   std::string("\n\tcoarsestSolver: ") + coarsest_solver_ +
                   std::string("\n\tcoarseMaxIters: ") +
                   std::to_string(coarse_max_iters_) +
                   std::string("\n\treductionFactor: ") +
                   std::to_string(inner_reduction_factor_) +
                   std::string("\n\tsmootherMaxIters: ") +
                   std::to_string(smoother_max_iters_) +
                   std::string("\n\tmaxLevels: ") +
                   std::to_string(max_levels_) +
                   std::string("\n\tminCoarseRows: ") +
                   std::to_string(min_coarse_rows_);
        MLOG_0(verbose_, msg)
    }

    std::shared_ptr<mg> create_dist_solver(
        std::shared_ptr<gko::Executor> exec,
        std::shared_ptr<gko::LinOp> sysmatrix, std::shared_ptr<dist_vec> x,
        std::shared_ptr<dist_vec> b, const label verbose, const bool export_res,
        std::shared_ptr<gko::LinOp> precond) const
    {
        gko::solver::multigrid::cycle cycle;
        if (cycle_ == "v") cycle = gko::solver::multigrid::cycle::v;
        if (cycle_ == "w") cycle = gko::solver::multigrid::cycle::w;
        if (cycle_ == "f") cycle = gko::solver::multigrid::cycle::f;

        auto gkomatrix =
            gko::as<RepartDistMatrix>(sysmatrix)->get_dist_matrix();

        auto gko_local_matrix =
            gko::as<
                gko::experimental::distributed::Matrix<scalar, label, label>>(
                gkomatrix)
                ->get_local_matrix();

        outerStoppingCriterionVec_.push_back(
            outerStoppingCriterion_.build_dist_stopping_criterion(
                exec, gkomatrix, x, b, verbose, export_res,
                get_prev_number_of_iterations(),
                get_solve_prev_rel_res_cost()));

        auto smoother_gen = gko::share(
            ir::build()
                .with_solver(ras::build().with_local_solver(
                    bj::build().with_skip_sorting(true).with_max_block_size(
                        1u)))
                .with_relaxation_factor(smoother_relaxation_factor_)
                .with_criteria(gko::stop::Iteration::build().with_max_iters(
                    smoother_max_iters_))
                .on(exec));

        // Create MultigridLevel factory
        auto mg_level_gen =
            pgm::build().with_deterministic(true).with_skip_sorting(true).on(
                exec);

        // Create CoarsestSolver factory
        std::shared_ptr<const gko::LinOpFactory> coarsest_solver{};

        // if (coarsest_solver_ == "CG") {
        coarsest_solver = gko::share(
            cg::build()
                .with_preconditioner(ras::build().with_local_solver(
                    bj::build().with_max_block_size(1u)))
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(
                        coarse_max_iters_),
                    gko::stop::ResidualNorm<scalar>::build()
                        .with_baseline(gko::stop::mode::absolute)
                        .with_reduction_factor(inner_reduction_factor_))
                .on(exec));

        // }

        // Create multigrid factory
        auto ret =
            mg::build()
                .with_max_levels(max_levels_)
                .with_mg_level(
                    gko::multigrid::Pgm<scalar>::build().with_deterministic(
                        false))
                .with_min_coarse_rows(min_coarse_rows_)
                .with_coarsest_solver(coarsest_solver)
                .with_pre_smoother(smoother_gen)
                .with_post_uses_pre(true)
                .with_criteria(outerStoppingCriterionVec_)
                .with_cycle(cycle)
                .on(exec);

        return gko::share(ret->generate(gkomatrix));
    }

    label get_res_norm_time() const
    {
        return outerStoppingCriterion_.get_res_norm_time();
    }

    scalar get_solve_prev_rel_res_cost() const
    {
        return ::Foam::get_solve_prev_rel_res_cost(sysMatrixName_, db_);
    }


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
                             outerStoppingCriterion_.get_num_iters(),
                             outerStoppingCriterion_.get_is_final());
    }

    void set_prev_rel_res_cost(scalar prev_rel_res_cost) const
    {
        return ::Foam::set_solve_prev_rel_res_cost(sysMatrixName_, db_,
                                                   prev_rel_res_cost);
    }

    label get_prev_number_of_iterations() const
    {
        return get_solve_prev_iters(sysMatrixName_, db_,
                                    outerStoppingCriterion_.get_is_final());
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
                           interfaceIntCoeffs, interfaces, solverControls)
    {}

    //- Destructor
    virtual ~GKOMultigrid() {}


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


// ************************************************************************* //
