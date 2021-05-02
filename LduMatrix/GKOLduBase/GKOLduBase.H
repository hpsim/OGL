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
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/

#ifndef GKO_LDUBase_H
#define GKO_LDUBase_H

#include "../IOExecutorHandler/IOExecutorHandler.H"
#include "../IOHandler/IOSortingIdxHandler/IOSortingIdxHandler.H"
#include "../common/StoppingCriterion.H"
#include "../common/common.H"
#include "LduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

/*---------------------------------------------------------------------------*\
                           Class GKOCG Declaration
\*---------------------------------------------------------------------------*/

template <class Type, class LDUType, class SolverFactory>
class GKOLduBaseSolver : public HostMatrix<LduMatrix<Type, LDUType, LDUType>>,
                         public SolverFactory,
                         public StoppingCriterion,
                         public IOSortingIdxHandler,
                         public IOGKOMatrixHandler {
public:
    // Some shortcuts
    using IndexType = label;
    using vec = gko::matrix::Dense<scalar>;
    using mtx = gko::matrix::Coo<scalar>;
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;
    using cg = gko::solver::Cg<scalar>;

private:
    // for now a copy of the OF matrix is stored
    // to keep values contiguous in memory
    // executor where Ginkgo will perform the computation

    std::vector<scalar> sorting_idxs_;

    mutable std::vector<scalar> values_;

    mutable std::vector<label> col_idxs_;

    mutable std::vector<label> row_idxs_;

public:
    //- Construct from matrix components and solver controls
    GKOLduBaseSolver(const word &fieldName,
                     const LduMatrix<Type, LDUType, LDUType> &matrix,
                     const dictionary &solverControls)
        : HostMatrix<LduMatrix<Type, LDUType, LDUType>>(fieldName, matrix,
                                                        solverControls),
          SolverFactory(solverControls),
          StoppingCriterion(solverControls),
          IOSortingIdxHandler(
              matrix.mesh().thisDb(), this->nElems(),
              solverControls.lookupOrDefault<Switch>("sort", true)),
          IOGKOMatrixHandler(matrix.mesh().thisDb(), solverControls, fieldName)
    {
        // if sys_matrix is not stored updating is neccesary
        // initially
        bool stored = get_sys_matrix_stored();
        if (!stored) {
            SIMPLE_TIME(init_host_sparsity_pattern,
                        this->init_host_sparsity_pattern();)
            std::cout << " matrix not stored update host matrix " << std::endl;
            SIMPLE_TIME(update_host_mtx, this->update_host_matrix_data();)
        } else {
            // if sys_matrix is  stored updating is only neccesary
            // when requested explictly
            std::cout
                << " matrix is stored and  update of host matrix requested"
                << std::endl;
            if (get_update_sys_matrix()) {
                SIMPLE_TIME(exp_update_host_mtx,
                            this->update_host_matrix_data();)
            }
        }

        // after updating the host matrix the host matrix needs to be sorted
        if (!get_is_sorted()) {
            std::cout << "matrix is not yet sorted, compute sorting idxs "
                      << std::endl;
            SIMPLE_TIME(compute_sort,
                        this->compute_sorting_idxs(this->row_idxs());)
        }

        if (!stored && get_sort()) {
            std::cout << "matrix is not yet sorted,  sorting host matrix "
                      << std::endl;
            SIMPLE_TIME(
                sort_host_mtx_sparsity_pattern,
                this->sort_host_matrix_sparsity_pattern(get_sorting_idxs());)
            SIMPLE_TIME(sort_host_mtx,
                        this->sort_host_matrix_data(this->get_sorting_idxs());)
        }

        init_device_matrix(matrix.mesh().thisDb(), this->values(),
                           this->col_idxs(), this->row_idxs(), this->nElems(),
                           this->nCells(), !get_update_sys_matrix());
    }


    virtual SolverPerformance<Type> solve_impl(Field<Type> &psi) const
    {
        std::shared_ptr<gko::Executor> device_exec =
            this->get_device_executor();
        // --- Setup class containing solver performance data
        // Implement
        word preconditionerName(this->controlDict_.lookup("preconditioner"));

        SolverPerformance<Type> solverPerf(preconditionerName,
                                           this->fieldName_);

        auto source_view_x = val_array::view(
            ref_exec(), 3 * this->nCells(),
            const_cast<scalar *>(&this->matrix().source()[0].x()));
        auto source_view_y = val_array::view(
            ref_exec(), 3 * this->nCells(),
            const_cast<scalar *>(&this->matrix().source()[0].y()));
        auto source_view_z = val_array::view(
            ref_exec(), 3 * this->nCells(),
            const_cast<scalar *>(&this->matrix().source()[0].z()));

        auto b_x = vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1),
                               source_view_x, 3);
        auto b_y = vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1),
                               source_view_y, 3);
        auto b_z = vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1),
                               source_view_z, 3);

        auto psi_view_x = val_array::view(ref_exec(), 3 * this->nCells(),
                                          const_cast<scalar *>(&psi[0].x()));
        auto psi_view_y = val_array::view(ref_exec(), 3 * this->nCells(),
                                          const_cast<scalar *>(&psi[0].y()));
        auto psi_view_z = val_array::view(ref_exec(), 3 * this->nCells(),
                                          const_cast<scalar *>(&psi[0].z()));

        auto x = vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1),
                             psi_view_x, 3);
        auto y = vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1),
                             psi_view_y, 3);
        auto z = vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1),
                             psi_view_z, 3);

        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec{};

        criterion_vec.push_back(build_stopping_criterion(device_exec, 1.0));

        // Generate solver
        auto solver_gen = this->create_solver(device_exec, criterion_vec);

        std::shared_ptr<mtx> gkomatrix = get_gkomatrix();

        auto solver = solver_gen->generate(gko::give(gkomatrix));

        SIMPLE_TIME(solve_x, solver->apply(gko::lend(b_x), gko::lend(x));)
        SIMPLE_TIME(solve_y, solver->apply(gko::lend(b_y), gko::lend(y));)
        SIMPLE_TIME(solve_z, solver->apply(gko::lend(b_z), gko::lend(z));)

        auto x_view =
            val_array::view(ref_exec(), this->nCells(), x->get_values());
        auto x_clone =
            vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1), x_view, 1);
        auto y_view =
            val_array::view(ref_exec(), this->nCells(), y->get_values());
        auto y_clone =
            vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1), y_view, 1);

        auto z_view =
            val_array::view(ref_exec(), this->nCells(), z->get_values());
        auto z_clone =
            vec::create(ref_exec(), gko::dim<2>(this->nCells(), 1), z_view, 1);

        for (size_t i = 0; i < this->nCells(); i++) {
            psi[i].x() = x_clone->at(i);
            psi[i].y() = y_clone->at(i);
            psi[i].z() = z_clone->at(i);
        }

        return solverPerf;
    };
};
}  // End namespace Foam

#endif
