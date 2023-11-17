/*---------------------------------------------------------------------------*\
License
    This file is part of OGL.

    OGL is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OGL is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OGL.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::GKOGMRES

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOGMRES.H

\*---------------------------------------------------------------------------*/

#ifndef GKOGMRES_H
#define GKOGMRES_H

#include "BaseWrapper/lduBase/GKOlduBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOGMRESFactory {
private:
    using mtx = gko::matrix::Csr<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    using gmres = gko::solver::Gmres<scalar>;
    using val_array = gko::array<scalar>;

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
    GKOGMRESFactory(const dictionary &solverControls, const objectRegistry &db,
                    word sysMatrixName)
        : solverControls_(solverControls),
          db_(db),
          sysMatrixName_(sysMatrixName),
          stoppingCriterion_(solverControls){};

    CREATE_SOLVER_METHODS(gmres)

    std::unique_ptr<gmres::Factory, std::default_delete<gmres::Factory>>
    create_default(std::shared_ptr<gko::Executor> exec) const
    {
        auto gmres = gmres::build()
                         .with_criteria(stoppingCriterionVec_)
                         .on(exec);
        return gmres;
    };

    std::unique_ptr<gmres::Factory, std::default_delete<gmres::Factory>>
    create_precond(std::shared_ptr<gko::Executor> exec,
                   std::shared_ptr<gko::LinOp> precond) const
    {
        auto gmres = gmres::build()
                         .with_criteria(stoppingCriterionVec_)
                         .with_generated_preconditioner(precond)
                         .on(exec);
        return gmres;
    };

    scalar get_init_res_norm() const
    {
        return stoppingCriterion_.get_init_res_norm();
    }

    scalar get_res_norm() const { return stoppingCriterion_.get_res_norm(); }

    scalar get_res_norm_time() const
    {
        return stoppingCriterion_.get_res_norm_time();
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
};

/*---------------------------------------------------------------------------*\
                           Class GKOGMRES Declaration
\*---------------------------------------------------------------------------*/


class GKOGMRES : public GKOlduBaseSolver<GKOGMRESFactory> {
    // Private Member Functions

public:
    TypeName("GKOGMRES");

    //- Disallow default bitwise copy construct
    GKOGMRES(const GKOGMRES &);

    //- Disallow default bitwise assignment
    void operator=(const GKOGMRES &);


    // Constructors

    //- Construct from matrix components and solver controls
    GKOGMRES(const word &fieldName, const lduMatrix &matrix,
             const FieldField<Field, scalar> &interfaceBouCoeffs,
             const FieldField<Field, scalar> &interfaceIntCoeffs,
             const lduInterfaceFieldPtrsList &interfaces,
             const dictionary &solverControls)
        : GKOlduBaseSolver(fieldName, matrix, interfaceBouCoeffs,
                           interfaceIntCoeffs, interfaces, solverControls){};

    //- Destructor
    virtual ~GKOGMRES(){};


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
