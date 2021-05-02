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
    along with OGL.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/

#ifndef GKOlduBase_H
#define GKOlduBase_H

#include "../IOHandler/IOExecutorHandler/IOExecutorHandler.H"
#include "../IOHandler/IOSortingIdxHandler/IOSortingIdxHandler.H"
#include "../common/StoppingCriterion.H"
#include "../common/common.H"

#include "IOPtrList.H"
#include "fvCFD.H"
#include "lduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

/*---------------------------------------------------------------------------*\
                           Class GKOlduBaseSolver Declaration
\*---------------------------------------------------------------------------*/


template <class SolverFactory>
class GKOlduBaseSolver : public HostMatrix<lduMatrix>,
                         public SolverFactory,
                         public StoppingCriterion,
                         public IOSortingIdxHandler,
                         public IOGKOMatrixHandler {
public:
    // Some shortcuts
    using IndexType = label;
    using vec = gko::matrix::Dense<scalar>;
    using idx_vec = gko::matrix::Dense<label>;
    using mtx = gko::matrix::Coo<scalar>;
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;
    using cg = gko::solver::Cg<scalar>;


    // for now a copy of the OF matrix is stored
    // to keep values contiguous in memory

public:
    //- Construct from matrix components and solver controls
    GKOlduBaseSolver(const word &fieldName, const lduMatrix &matrix,
                     const FieldField<Field, scalar> &interfaceBouCoeffs,
                     const FieldField<Field, scalar> &interfaceIntCoeffs,
                     const lduInterfaceFieldPtrsList &interfaces,
                     const dictionary &solverControls)
        : HostMatrix<lduMatrix>(fieldName, matrix, interfaceBouCoeffs,
                                interfaceIntCoeffs, interfaces, solverControls),
          SolverFactory(solverControls),
          StoppingCriterion(solverControls),
          IOSortingIdxHandler(
              matrix.mesh().thisDb(), nElems(),
              solverControls.lookupOrDefault<Switch>("sort", true)),
          IOGKOMatrixHandler(matrix.mesh().thisDb(), solverControls, fieldName)
    {
        // if sys_matrix is not stored updating is neccesary
        // initially
        bool stored = get_sys_matrix_stored();
        if (!stored) {
            SIMPLE_TIME(init_host_sparsity_pattern,
                        init_host_sparsity_pattern();)
            SIMPLE_TIME(update_host_mtx, update_host_matrix_data();)
        } else {
            // if sys_matrix is  stored updating is only neccesary
            // when requested explictly
            if (get_update_sys_matrix()) {
                // since sparsity pattern should already be stored
                // on device no init_host_sparsity_pattern call is needed
                SIMPLE_TIME(exp_update_host_mtx, update_host_matrix_data();)
            }
        }

        // TODO move compute_sorting_idxs in the if clause above to have all
        // init calls together
        // after updating the host matrix the host matrix needs to be sorted
        if (!get_is_sorted()) {
            SIMPLE_TIME(compute_sorting_idxs,
                        compute_sorting_idxs(row_idxs(), col_idxs(), nCells());)
        }

        // TODO move compute_sorting_idxs in the if clause above to have all
        // init calls together
        if (!stored && get_sort()) {
            SIMPLE_TIME(sort_host_mtx_sparsity_pattern,
                        sort_host_matrix_sparsity_pattern(get_sorting_idxs());)
            SIMPLE_TIME(sort_host_mtx_data,
                        sort_host_matrix_data(get_sorting_idxs());)
        }

        init_device_matrix(matrix.mesh().thisDb(), values(), col_idxs(),
                           row_idxs(), nElems(), nCells(),
                           !get_update_sys_matrix());
    };

    virtual solverPerformance solve_impl(word typeName, scalarField &psi,
                                         const scalarField &source,
                                         const direction cmpt = 0) const
    {
        std::shared_ptr<gko::Executor> device_exec =
            this->get_device_executor();

        // --- Setup class containing solver performance data
        solverPerformance solverPerf(
            lduMatrix::preconditioner::getName(controlDict_) + typeName,
            fieldName_);

        scalarField pA(nCells());
        scalarField wA(nCells());
        matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
        scalar norm_factor = this->normFactor(psi, source, wA, pA);

        auto source_view = val_array::view(ref_exec(), nCells(),
                                           const_cast<scalar *>(&source[0]));
        auto b =
            vec::create(ref_exec(), gko::dim<2>(nCells(), 1), source_view, 1);

        std::cout << "solve_impl init_initial_guess" << std::endl;
        init_initial_guess(psi, matrix().mesh().thisDb(), nCells());
        std::cout << "solve_impl init_initial_guess done" << std::endl;
        auto x = this->get_initial_guess();
        std::cout << "solve_impl get_initial_guess done" << std::endl;

        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec{};

        criterion_vec.push_back(
            build_stopping_criterion(device_exec, norm_factor));

        // Generate solver
        auto solver_gen = this->create_solver(device_exec, criterion_vec);

        // Instantiate a ResidualLogger logger.
        auto logger = std::make_shared<IterationLogger>(device_exec);

        // Add the previously created logger to the solver factory. The
        // logger will be automatically propagated to all solvers created
        // from this factory.
        solver_gen->add_logger(logger);

        // auto b_clone = gko::clone(b);

        std::shared_ptr<mtx> gkomatrix = get_gkomatrix();

        if (get_export())
            export_system(fieldName(), gko::lend(gkomatrix), gko::lend(x),
                          gko::lend(b));


        solverPerf.initialResidual() = 0.0;

        auto solver = solver_gen->generate(gko::share(gkomatrix));

        // Solve system
        SIMPLE_TIME(solve, solver->apply(gko::lend(b), gko::lend(x));)

        this->copy_result_back(psi, nCells());

        // b_clone->copy_from(gko::lend(b));

        solverPerf.finalResidual() = 0.0;

        solverPerf.nIterations() = logger->get_iters();

        return solverPerf;
    };
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// //

}  // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
