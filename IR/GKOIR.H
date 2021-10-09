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
    Foam::GKOIR

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOIR.C

\*---------------------------------------------------------------------------*/

#ifndef GKOIR_H
#define GKOIR_H

#include "../BaseWrapper/lduBase/GKOlduBase.H"
#include "../common/StoppingCriterion.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

class GKOIRFactory {
private:
    using ir = gko::solver::Ir<scalar>;
    // executor where Ginkgo will perform the computation
    const dictionary &solverControls_;

    const StoppingCriterion outerStoppingCriterion_;

    mutable std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        outerStoppingCriterionVec_;

public:
    GKOIRFactory(const dictionary &solverControls)
        : solverControls_(solverControls),
          outerStoppingCriterion_(solverControls){};

    std::unique_ptr<ir::Factory, std::default_delete<ir::Factory>>
    create_solver(std::shared_ptr<gko::Executor> exec,
                  std::shared_ptr<gko::log::Logger> logger,
                  std::shared_ptr<mtx> gkomatrix, std::shared_ptr<vec> x,
                  std::shared_ptr<vec> b, label prev_solve_iters,
                  const bool verbose, std::shared_ptr<gko::LinOp> precond) const

    {
        outerStoppingCriterionVec_.push_back(
            outerStoppingCriterion_.build_stopping_criterion(
                exec, gkomatrix, x, b, verbose, prev_solve_iters));

        // if (solverinner_solver_ == "scalarJacobi") {
        //     return create_scalar_jacobi(exec, gkomatrix, x, b,
        //     prev_solve_iters,
        //                                 logger, verbose);
        // }
        return create_default(exec, gkomatrix, x, b, prev_solve_iters, logger,
                              verbose);
    };

    std::unique_ptr<ir::Factory,
                    std::default_delete<gko::solver::Ir<double>::Factory>>
    create_default(std::shared_ptr<gko::Executor> exec,
                   std::shared_ptr<mtx> gkomatrix, std::shared_ptr<vec> x,
                   std::shared_ptr<vec> b, label prev_solve_iters,
                   std::shared_ptr<gko::log::Logger> logger,
                   const bool verbose) const
    {
        auto innerStoppingCriterionFactory =
            StoppingCriterion(solverControls_.subDict("inner"));

        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            innerStoppingCriterion{};

        innerStoppingCriterion.push_back(
            innerStoppingCriterionFactory.build_stopping_criterion(
                exec, gkomatrix, x, b, verbose, prev_solve_iters));


        auto inner = gko::share(gko::solver::Cg<scalar>::build()
                                    .with_criteria(innerStoppingCriterion)
                                    .on(exec));

        inner->add_logger(logger);

        auto ir = ir::build()
                      .with_solver(inner)
                      .with_criteria(outerStoppingCriterionVec_)
                      .on(exec);

        ir->add_logger(logger);
        return ir;
    };

    std::unique_ptr<gko::solver::Ir<double>::Factory,
                    std::default_delete<gko::solver::Ir<double>::Factory>>
    create_scalar_jacobi(
        std::shared_ptr<gko::Executor> exec,
        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            outerStoppingCriterion,
        std::shared_ptr<mtx> gkomatrix, std::shared_ptr<vec> x,
        std::shared_ptr<vec> b, label prev_solve_iters,
        std::shared_ptr<gko::log::Logger> logger, const bool verbose) const
    {
        int sweeps = 3;
        using bj = gko::preconditioner::Jacobi<>;
        label blockSize = solverControls_.subDict("inner").lookupOrDefault(
            "maxBlockSize", label(4));
        auto bj_factory = bj::build()
                              .with_max_block_size(blockSize)
                              // .with_storage_optimization(
                              //     gko::precision_reduction::autodetect())
                              .on(exec);

        auto ir = ir::build()
                      .with_solver(gko::share(bj_factory))
                      .with_criteria(outerStoppingCriterionVec_)
                      .on(exec);

        ir->add_logger(logger);
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
