// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "BaseWrapper/lduBase/GKOlduBase.H"
#include "StoppingCriterion/StoppingCriterion.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOMultigridFactory {
private:
    using cg = gko::solver::Cg<scalar>;
    using direct = gko::experimental::solver::Direct<scalar, label>;
    using mtx = gko::matrix::Csr<scalar>;
    using ir = gko::solver::Ir<scalar>;
    using mg = gko::solver::Multigrid;
    using bj = gko::preconditioner::Jacobi<scalar, label>;
    using amgx_pgm = gko::multigrid::AmgxPgm<scalar, label>;
    using ras =
        gko::experimental::distributed::preconditioner::Schwarz<scalar, label,
                                                                label>;

    using vec = gko::matrix::Dense<scalar>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;

    const dictionary &solverControls_;

    const objectRegistry &db_;

    const word sysMatrixName_;

    const StoppingCriterion outerStoppingCriterion_;

    const label verbose_;

    const word coarsest_solver_;

    const scalar inner_relaxation_factor_;

    const scalar smoother_relaxation_factor_;

    const label smoother_max_iters_;

    const label coarse_max_iters_;

    const label max_levels_;

    const label min_coarse_rows_;

    const label cycle_;

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        outerStoppingCriterionVec_;

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        innerStoppingCriterionVec_;

public:
    GKOMultigridFactory(const dictionary &solverControls,
                        const objectRegistry &db, word sysMatrixName)
        : solverControls_(solverControls),
          db_(db),
          sysMatrixName_(sysMatrixName),
          outerStoppingCriterion_(solverControls),
          verbose_(solverControls.lookupOrDefault<label>("verbose", 0)),
          coarsest_solver_(
              solverControls_.lookupOrDefault("coarsestSolver", word("CG"))),
          inner_relaxation_factor_(solverControls_.lookupOrDefault(
              "innerRelaxationFactor", scalar(0.9))),
          smoother_relaxation_factor_(solverControls_.lookupOrDefault(
              "smootherRelaxationFactor", scalar(0.9))),
          smoother_max_iters_(
              solverControls_.lookupOrDefault("smootherMaxIters", label(2))),
          coarse_max_iters_(
              solverControls_.lookupOrDefault("coarseMaxIters", label(4))),
          max_levels_(solverControls_.lookupOrDefault("maxLevels", label(9))),
          min_coarse_rows_(
              solverControls_.lookupOrDefault("minCoarseRows", label(10))),
          cycle_(solverControls_.lookupOrDefault("cycle", label(0)))
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
                   std::string("\n\tsmootherMaxIters: ") +
                   std::to_string(smoother_max_iters_) +
                   std::string("\n\tmaxLevels: ") +
                   std::to_string(max_levels_) +
                   std::string("\n\tminCoasresRows: ") +
                   std::to_string(min_coarse_rows_);
        MLOG_0(verbose_, msg)
    }

    std::unique_ptr<mg::Factory, std::default_delete<mg::Factory>>
    create_dist_solver(std::shared_ptr<const gko::Executor> exec,
                       std::shared_ptr<const gko::LinOp> gkomatrix,
                       std::shared_ptr<const dist_vec> x,
                       std::shared_ptr<const dist_vec> b, const label verbose,
                       const bool export_res,
                       std::shared_ptr<gko::LinOp> precond) const
    {
        outerStoppingCriterionVec_.push_back(
            outerStoppingCriterion_.build_dist_stopping_criterion(
                exec, gkomatrix, x, b, verbose, export_res,
                get_prev_number_of_iterations(),
                get_solve_prev_rel_res_cost()));

        std::shared_ptr<const gko::LinOpFactory> coarsest_solver{};

        if (coarsest_solver_ == "CG") {
            coarsest_solver = gko::share(
                cg::build()
                    // .with_preconditioner(ras::build().with_local_solver(
                    //     bj::build().with_max_block_size(1u)))
                    .with_criteria(gko::stop::Iteration::build().with_max_iters(
                                       coarse_max_iters_),
                                   gko::stop::ResidualNorm<scalar>::build()
                                       .with_baseline(gko::stop::mode::absolute)
                                       .with_reduction_factor(1e-18))
                    .on(exec));
        } else {
            FatalErrorInFunction
                << "Unsupported coarsestSolver: " << coarsest_solver_
                << "! Only CG is supported" << abort(FatalError);
        }
        // if (coarsest_solver_ == "direct") {
        //     coarsest_solver = gko::share(
        //         ras::build()
        //             .with_local_solver(direct::build().with_factorization(
        //                 gko::experimental::factorization::Lu<scalar,
        //                                                      label>::build()
        //                     .with_symbolic_algorithm(
        //                         gko::experimental::factorization::
        //                             symbolic_type::symmetric)))
        //             .on(exec));
        // }

        auto ret =
            gko::solver::Multigrid::build()
                .with_max_levels(max_levels_)
                .with_mg_level(
                    gko::multigrid::Pgm<scalar>::build().with_deterministic(
                        false))
                .with_min_coarse_rows(min_coarse_rows_)
                .with_coarsest_solver(coarsest_solver)
                .with_criteria(outerStoppingCriterionVec_)
                .with_smoother_iters(smoother_max_iters_)
                .with_cycle(cycle_)
                // .with_pre_smoother(
                //         smoother_gen
                //         )
                // .with_post_uses_pre(true)
                .on(exec);
        return ret;
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

    void set_prev_rel_res_cost(scalar prev_rel_res_cost) const
    {
        return ::Foam::set_solve_prev_rel_res_cost(sysMatrixName_, db_,
                                                   prev_rel_res_cost);
    }

    void store_number_of_iterations() const
    {
        set_solve_prev_iters(sysMatrixName_, db_,
                             outerStoppingCriterion_.get_num_iters(),
                             outerStoppingCriterion_.get_is_final());
    }

    label get_prev_number_of_iterations() const
    {
        return 0;
        // get_solve_prev_iters(sysMatrixName_, db_,
        //                             innerStoppingCriterion_.get_is_final());
    }

    label get_res_norm_time() const
    {
        return outerStoppingCriterion_.get_res_norm_time();
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
