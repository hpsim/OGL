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

#ifndef GKOBase_H
#define GKOBase_H

#define SIMPLE_TIME(NAME, F)  \
    auto start_##NAME = std::chrono::steady_clock::now() ; \
    F \
    auto end_##NAME = std::chrono::steady_clock::now(); \
    std::cout <<  "Gingko " #NAME " : " << \
        std::chrono::duration_cast< \
        std::chrono::milliseconds>(end_##NAME - start_##NAME).count() << " ms\n";


#include "lduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Logs the number of iteration executed
struct IterationLogger : gko::log::Logger {
    void on_iteration_complete(const gko::LinOp *,
                                const gko::size_type &num_iterations,
                                const gko::LinOp *, const gko::LinOp *,
                                const gko::LinOp *) const override
    {
        this->num_iters = num_iterations;
    }

    IterationLogger(std::shared_ptr<const gko::Executor> exec)
        : gko::log::Logger(exec, gko::log::Logger::iteration_complete_mask)
    {}

    gko::size_type get_iters() {return num_iters;}

private:
    mutable gko::size_type num_iters{0};
};

/*---------------------------------------------------------------------------*\
                           Class GKOCG Declaration
\*---------------------------------------------------------------------------*/

class GKOBaseSolver :
    public lduMatrix::solver
{
public:
    // Some shortcuts
    using IndexType = label;
    using vec = gko::matrix::Dense<scalar>;
    using mtx = gko::matrix::Coo<scalar>;
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;
    using cg = gko::solver::Cg<scalar>;

private:

    // executor where Ginkgo will perform the computation
    std::shared_ptr<gko::Executor> exec_;

    // executor of the application
    std::shared_ptr<gko::Executor> app_exec_;

    label nCells_;

    label nNeighbours_;

    label nElems_;

    // for now a copy of the OF matrix is stored
    // to keep values contiguous in memory
public:

    std::vector<scalar> sorting_idxs_;

    mutable std::vector<scalar> values_;

    mutable std::vector<label> col_idxs_;

    mutable std::vector<label> row_idxs_;

    // std::unique_ptr<
    //     gko::matrix::Coo<scalar>,
    //     std::default_delete<gko::matrix::Coo<scalar> >
    // > gkomatrix_;




public:
    //- Construct from matrix components and solver controls
    GKOBaseSolver
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
        );

    void compute_sorting_idxs();

    std::shared_ptr<gko::Executor> exec() const { return exec_;};

    std::shared_ptr<gko::Executor> app_exec() const { return app_exec_;};

    const label nCells() const {return nCells_;};

    const label nElems() const {return nElems_;};

    const std::vector<scalar>& vals() const {return values_;};

    const std::vector<label>& col_idx() const {return col_idxs_;};

    const std::vector<label>& row_idx() const {return row_idxs_;};

    void update_GKOMatrix() const;

    void sort_GKOMatrix() const;

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
