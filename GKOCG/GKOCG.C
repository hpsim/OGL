/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>
#include "GKOCG.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(GKOCG, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<GKOCG>
        addGKOCGSymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::GKOCG::GKOCG
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    GKOBaseSolver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{



}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::GKOCG::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );
    solverPerf.initialResidual() = 1.0;

    auto b = vec::create(
        exec(),
        gko::dim<2>(nCells(), 1),
        val_array::view(app_exec(), nCells(), const_cast<scalar *>(&source[0])),
        1);

    auto x = vec::create(
        exec(),
        gko::dim<2>(nCells(), 1),
        val_array::view(app_exec(), nCells(), &psi[0]),
        1);

    SIMPLE_TIME(update_matrix, update_GKOMatrix();)

    SIMPLE_TIME(sort, sort_GKOMatrix();)

    // Generate solver
    auto solver_gen =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(maxIter_).on(exec()),
                    gko::stop::ResidualNormReduction<scalar>::build()
                    .with_reduction_factor(tolerance_)
                .on(exec()))
        .on(exec());

    // Instantiate a ResidualLogger logger.
    auto logger = std::make_shared<IterationLogger>(exec());

    // Add the previously created logger to the solver factory. The logger will
    // be automatically propagated to all solvers created from this factory.
    solver_gen->add_logger(logger);

    auto gkomatrix = gko::give(mtx::create(exec(),
        gko::dim<2>(nCells()),
        val_array::view(app_exec(), nElems(), &values_[0]),
        idx_array::view(app_exec(), nElems(), &col_idxs_[0]),
        idx_array::view(app_exec(), nElems(), &row_idxs_[0])));

    auto solver = solver_gen->generate(gko::give(gkomatrix));

    // Solve system
    SIMPLE_TIME(solve, solver->apply(gko::lend(b), gko::lend(x)); )

    auto one = gko::initialize<vec>({1.0}, exec());
    auto neg_one = gko::initialize<vec>({-1.0}, exec());
    auto res = gko::initialize<vec>({0.0}, exec());
    b->compute_norm2(lend(res));

    solverPerf.finalResidual() =  1.0;
    solverPerf.nIterations() =  logger->get_iters();

    return solverPerf;
}


// ************************************************************************* //
