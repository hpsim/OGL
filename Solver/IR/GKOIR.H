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
    Foam::GKOIR

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOIR.C

\*---------------------------------------------------------------------------*/

#ifndef GKOIR_H
#define GKOIR_H

#include "BaseWrapper/lduBase/GKOlduBase.H"
#include "StoppingCriterion/StoppingCriterion.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOIRFactory {
private:
    using ir = gko::solver::Ir<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    using cg = gko::solver::Cg<scalar>;
    using val_array = gko::array<scalar>;

    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;

    const dictionary &solverControls_;

    const dictionary &innerSolverControls_;

    const objectRegistry &db_;

    const word sysMatrixName_;

    const StoppingCriterion outerStoppingCriterion_;

    const StoppingCriterion innerStoppingCriterion_;

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        outerStoppingCriterionVec_ = {};

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        innerStoppingCriterionVec_ = {};

public:
    GKOIRFactory(const dictionary &solverControls, const objectRegistry &db,
                 word sysMatrixName)
        : solverControls_(solverControls),
          innerSolverControls_(solverControls.subDict("inner")),
          db_(db),
          sysMatrixName_(sysMatrixName),
          outerStoppingCriterion_(solverControls),
          innerStoppingCriterion_(StoppingCriterion(innerSolverControls_)){};


    std::unique_ptr<ir::Factory, std::default_delete<ir::Factory>>  
    create_dist_solver(                                                     
        std::shared_ptr<gko::Executor> exec,                                
        std::shared_ptr<gko::LinOp> gkomatrix, std::shared_ptr<dist_vec> x, 
        std::shared_ptr<dist_vec> b, const label verbose,                   
        const bool export_res, std::shared_ptr<gko::LinOp> precond) const   
    {                                                                       
        outerStoppingCriterionVec_.push_back(                                    
            outerStoppingCriterion_.build_dist_stopping_criterion(               
                exec, gkomatrix, x, b, verbose, export_res,                 
                get_prev_number_of_iterations(),                            
                get_solve_prev_rel_res_cost()));                            
                                                                            
        return create_default(exec);                                        
    };


    std::unique_ptr<ir::Factory,
                    std::default_delete<gko::solver::Ir<scalar>::Factory>>
    create_default(std::shared_ptr<gko::Executor> exec) const
    {
        auto inner = gko::share(gko::solver::Cg<scalar>::build()
                                    .with_criteria(innerStoppingCriterionVec_)
                                    .on(exec));

        auto ir = ir::build()
                      .with_solver(inner)
                      .with_criteria(outerStoppingCriterionVec_)
                      .on(exec);

        return ir;
    };


    scalar get_init_res_norm() const
    {
        return outerStoppingCriterion_.get_init_res_norm();
    }

    scalar get_res_norm() const
    {
        return outerStoppingCriterion_.get_res_norm();
    }

    scalar get_res_norm_time() const
    {
        return outerStoppingCriterion_.get_res_norm_time();
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

    label get_prev_number_of_iterations() const
    {
        return get_solve_prev_iters(sysMatrixName_, db_,
                                    outerStoppingCriterion_.get_is_final());
    }

    label get_number_of_iterations() const
    {
        return outerStoppingCriterion_.get_num_iters();
    }

    // TODO FIXME add via macro or base class
    void set_prev_rel_res_cost(scalar prev_rel_res_cost) const
    {
        return ::Foam::set_solve_prev_rel_res_cost(sysMatrixName_, db_,
                                                   prev_rel_res_cost);
    }

    scalar get_solve_prev_rel_res_cost() const
    {
        return ::Foam::get_solve_prev_rel_res_cost(sysMatrixName_, db_);
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
