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
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/

#ifndef GKOCG_H
#define GKOCG_H

#include "../BaseWrapper/lduBase/GKOlduBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOCGFactory {
private:
    using mtx = gko::matrix::Csr<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    using cg = gko::solver::Cg<scalar>;

    using dist_vec = gko::distributed::Vector<scalar>;
    using dist_mtx = gko::distributed::Matrix<scalar, label, label>;

    const dictionary &solverControls_;

    const objectRegistry &db_;

    const word sysMatrixName_;

    const StoppingCriterion stoppingCriterion_;

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        stoppingCriterionVec_;

public:
    GKOCGFactory(const dictionary &solverControls, const objectRegistry &db,
                 word sysMatrixName)
        : solverControls_(solverControls),
          db_(db),
          sysMatrixName_(sysMatrixName),
          stoppingCriterion_(solverControls){};

    CREATE_SOLVER_METHODS(cg)

    std::unique_ptr<cg::Factory, std::default_delete<cg::Factory>>
    create_default(std::shared_ptr<gko::Executor> exec) const
    {
        auto cg = cg::build().with_criteria(stoppingCriterionVec_).on(exec);
        return cg;
    };

    std::unique_ptr<cg::Factory, std::default_delete<cg::Factory>>
    create_precond(std::shared_ptr<gko::Executor> exec,
                   std::shared_ptr<gko::LinOp> precond) const
    {
        auto cg = cg::build()
                      .with_criteria(stoppingCriterionVec_)
                      .with_generated_preconditioner(precond)
                      .on(exec);
        return cg;
    };

    scalar get_init_res_norm() const
    {
        return stoppingCriterion_.get_init_res_norm();
    }

    scalar get_res_norm() const { return stoppingCriterion_.get_res_norm(); }

    scalar get_res_norm_time() const { return stoppingCriterion_.get_res_norm_time(); }

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
                           Class GKOCG Declaration
\*---------------------------------------------------------------------------*/


class GKOCG : public GKOlduBaseSolver<GKOCGFactory> {
    // Private Member Functions

public:
    TypeName("GKOCG");

    //- Disallow default bitwise copy construct
    GKOCG(const GKOCG &);

    //- Disallow default bitwise assignment
    void operator=(const GKOCG &);


    // Constructors

    //- Construct from matrix components and solver controls
    GKOCG(const word &fieldName, const lduMatrix &matrix,
          const FieldField<Field, scalar> &interfaceBouCoeffs,
          const FieldField<Field, scalar> &interfaceIntCoeffs,
          const lduInterfaceFieldPtrsList &interfaces,
          const dictionary &solverControls)
        : GKOlduBaseSolver(fieldName, matrix, interfaceBouCoeffs,
                           interfaceIntCoeffs, interfaces, solverControls){};

    //- Destructor
    virtual ~GKOCG(){};


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
