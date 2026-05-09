// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "OGL/GKOlduBase.hpp"
#include "OGL/StoppingCriterion.hpp"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOIRFactory {
private:
    using ir = gko::solver::Ir<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    using mtx = gko::matrix::Csr<scalar>;
    using cg = gko::solver::Cg<scalar>;
    using val_array = gko::array<scalar>;

    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;

    const dictionary &solverControls_;

    const dictionary &innerSolverControls_;

    const objectRegistry &db_;

    const word sysMatrixName_;

    const StoppingCriterion stoppingCriterion_;

    const StoppingCriterion innerStoppingCriterion_;

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        stoppingCriterionVec_ = {};

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        innerStoppingCriterionVec_ = {};

public:
    GKOIRFactory(const dictionary &solverControls, const objectRegistry &db,
                 word sysMatrixName)
        : solverControls_(solverControls),
          innerSolverControls_(solverControls.subDict("inner")),
          db_(db),
          sysMatrixName_(sysMatrixName),
          stoppingCriterion_(solverControls),
          innerStoppingCriterion_(StoppingCriterion(innerSolverControls_))
    {}

    std::shared_ptr<ir> create_dist_solver(
        std::shared_ptr<gko::Executor> exec,
        std::shared_ptr<gko::LinOp> gkomatrix, std::shared_ptr<dist_vec> x,
        std::shared_ptr<dist_vec> b, const label verbose,
        const bool export_res, std::shared_ptr<gko::LinOp> precond) const
    {
        stoppingCriterionVec_.push_back(
            stoppingCriterion_.build_dist_stopping_criterion(
                exec, gkomatrix, x, b, verbose, export_res,
                get_prev_number_of_iterations(),
                get_solve_prev_rel_res_cost()));

        innerStoppingCriterionVec_.push_back(
            innerStoppingCriterion_.build_dist_stopping_criterion(
                exec, gkomatrix, x, b, verbose, export_res, 0, 0));

        if (precond != NULL) return create_precond(exec, precond, gkomatrix);
        return create_default(exec, gkomatrix);
    }

    std::shared_ptr<ir> create_default(std::shared_ptr<gko::Executor> exec,
                                       std::shared_ptr<gko::LinOp> gkomatrix
    ) const
    {
        auto inner = gko::share(cg::build()
                                    .with_criteria(innerStoppingCriterionVec_)
                                    .on(exec));

        auto ir_solver = ir::build()
                             .with_solver(inner)
                             .with_criteria(stoppingCriterionVec_)
                             .on(exec);

        return gko::share(ir_solver->generate(gkomatrix));
    }

    std::shared_ptr<ir> create_precond(
        std::shared_ptr<gko::Executor> exec,
        std::shared_ptr<gko::LinOp> precond,
        std::shared_ptr<gko::LinOp> gkomatrix) const
    {
        auto inner = gko::share(cg::build()
                                    .with_criteria(innerStoppingCriterionVec_)
                                    .on(exec));

        auto ir_solver = ir::build()
                             .with_generated_solver(precond)
                             .with_criteria(stoppingCriterionVec_)
                             .on(exec);

        return gko::share(ir_solver->generate(gkomatrix));
    }

    scalar get_init_res_norm() const
    {
        return stoppingCriterion_.get_init_res_norm();
    }

    scalar get_res_norm() const { return stoppingCriterion_.get_res_norm(); }

    scalar get_res_norm_time() const
    {
        return stoppingCriterion_.get_res_norm_time();
    }

    std::shared_ptr<vec> get_res_norms() const
    {
        return stoppingCriterion_.get_res_norms();
    }

    void store_number_of_iterations() const
    {
        set_solve_prev_iters(sysMatrixName_, db_,
                             stoppingCriterion_.get_num_iters(),
                             stoppingCriterion_.get_is_final());
    }

    label get_prev_number_of_iterations() const
    {
        return get_solve_prev_iters(sysMatrixName_, db_,
                                    stoppingCriterion_.get_is_final());
    }

    label get_number_of_iterations() const
    {
        return stoppingCriterion_.get_num_iters();
    }

    scalar get_solve_prev_rel_res_cost() const
    {
        return ::Foam::get_solve_prev_rel_res_cost(sysMatrixName_, db_);
    }

    void set_prev_rel_res_cost(scalar prev_rel_res_cost) const
    {
        return ::Foam::set_solve_prev_rel_res_cost(sysMatrixName_, db_,
                                                   prev_rel_res_cost);
    }
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
                           interfaceIntCoeffs, interfaces, solverControls)
    {}

    //- Destructor
    virtual ~GKOIR() {}


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
