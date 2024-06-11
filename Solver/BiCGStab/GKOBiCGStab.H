// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "BaseWrapper/lduBase/GKOlduBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

/*---------------------------------------------------------------------------*\
                           Class GKOBiCGStab Declaration
\*---------------------------------------------------------------------------*/
class GKOBiCGStabFactory {
private:
    using mtx = gko::matrix::Csr<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    using bicgstab = gko::solver::Bicgstab<scalar>;

    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;

    const dictionary &solverControls_;

    const objectRegistry &db_;

    const word sysMatrixName_;

    const StoppingCriterion stoppingCriterion_;

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        stoppingCriterionVec_;


public:
    GKOBiCGStabFactory(const dictionary &solverControls,
                       const objectRegistry &db, word sysMatrixName)
        : solverControls_(solverControls),
          db_(db),
          sysMatrixName_(sysMatrixName),
          stoppingCriterion_(solverControls)
    {}

    CREATE_SOLVER_METHODS(bicgstab)

    std::unique_ptr<bicgstab::Factory, std::default_delete<bicgstab::Factory>>
    create_default(std::shared_ptr<gko::Executor> exec) const
    {
        auto bicgstab = gko::solver::Bicgstab<scalar>::build()
                            .with_criteria(stoppingCriterionVec_)
                            .on(exec);
        return bicgstab;
    }

    std::unique_ptr<bicgstab::Factory, std::default_delete<bicgstab::Factory>>
    create_precond(std::shared_ptr<gko::Executor> exec,
                   std::shared_ptr<gko::LinOp> precond) const
    {
        auto bicgstab = bicgstab::build()
                            .with_criteria(stoppingCriterionVec_)
                            .with_generated_preconditioner(precond)
                            .on(exec);
        return bicgstab;
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

    scalar get_solve_prev_rel_res_cost() const
    {
        return ::Foam::get_solve_prev_rel_res_cost(sysMatrixName_, db_);
    }

    void set_prev_rel_res_cost(scalar prev_rel_res_cost) const
    {
        return ::Foam::set_solve_prev_rel_res_cost(sysMatrixName_, db_,
                                                   prev_rel_res_cost);
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
        return stoppingCriterion_.get_num_iters() / 2;
    }
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
    virtual ~GKOBiCGStab() {}


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
