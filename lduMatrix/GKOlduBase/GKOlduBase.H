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

#ifndef GKOlduBase_H
#define GKOlduBase_H
#include "../common/StoppingCriterion.H"
#include "../common/common.H"

#include "IOPtrList.H"
#include "fvCFD.H"
#include "lduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

/*---------------------------------------------------------------------------*\
                           Class GKOCG Declaration
\*---------------------------------------------------------------------------*/

template <class SolverFactory>
class GKOlduBaseSolver : public lduMatrix::solver,
                         public SolverFactory,
                         public lduLduCommonBase {
public:
    // Some shortcuts
    using IndexType = label;
    using vec = gko::matrix::Dense<scalar>;
    using mtx = gko::matrix::Coo<scalar>;
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;
    using cg = gko::solver::Cg<scalar>;

    IOField<label> *sorting_idxs_ = NULL;

    mutable GKOCOOIOPtr *gkomatrix_ptr = NULL;

    // for now a copy of the OF matrix is stored
    // to keep values contiguous in memory
public:
    mutable std::vector<scalar> values_;

    mutable std::vector<label> col_idxs_;

    mutable std::vector<label> row_idxs_;

public:
    //- Construct from matrix components and solver controls
    GKOlduBaseSolver(const word &fieldName, const lduMatrix &matrix,
                     const FieldField<Field, scalar> &interfaceBouCoeffs,
                     const FieldField<Field, scalar> &interfaceIntCoeffs,
                     const lduInterfaceFieldPtrsList &interfaces,
                     const dictionary &solverControls)
        : lduMatrix::solver(fieldName, matrix, interfaceBouCoeffs,
                            interfaceIntCoeffs, interfaces, solverControls),
          SolverFactory(controlDict_),
          lduLduCommonBase(matrix.mesh().thisDb(), matrix.diag().size(),
                           matrix.lduAddr().upperAddr().size(), solverControls)

    {
        std::shared_ptr<gko::Executor> device_exec{};
        set_device_executor(device_exec);

        // TODO refactor to separate function
        // check object registry if sorting idxs exists
        word sorting_idxs_name = "sorting_idxs_";
        const objectRegistry &db = get_db();
        if (db.foundObject<IOField<label>>(sorting_idxs_name)) {
            IOField<label> &labelListRef =
                db.lookupObjectRef<IOField<label>>(sorting_idxs_name);
            sorting_idxs_ = &labelListRef;
            set_is_sorted(true);
        } else {
            const fileName path = sorting_idxs_name;
            sorting_idxs_ = new IOField<label>(IOobject(path, db));
            init_sorting_idxs();
            set_is_sorted(false);
        }


        // TODO use different matrix names
        word sys_matrix_name = "gko_sys_matrix";
        bool sys_matrix_stored = db.foundObject<regIOobject>(sys_matrix_name);

        if (sys_matrix_stored) {
            gkomatrix_ptr = &db.lookupObjectRef<GKOCOOIOPtr>(sys_matrix_name);
            return;
        }

        // if sys_matrix is not stored updating is only neccesary
        // when requested explictly or initially
        if (get_update_sys_matrix()) {
            SIMPLE_TIME(update_gko_mtx, update_GKOMatrix();)
        }

        if (!get_update_sys_matrix() && !sys_matrix_stored) {
            SIMPLE_TIME(update_gko_mtx, update_GKOMatrix();)
        }

        if (!get_is_sorted()) {
            SIMPLE_TIME(compute_sort, compute_sorting_idxs();)
        }

        if (get_sort()) {
            SIMPLE_TIME(sort_gko_mtx, sort_GKOMatrix();)
        }

        // if system matrix is not stored create it
        // and set shared pointer
        //
        auto gkomatrix = gko::share(
            mtx::create(device_exec, gko::dim<2>(nCells(), nCells()),
                        val_array::view(gko::ReferenceExecutor::create(),
                                        nElems(), &values_[0]),
                        idx_array::view(gko::ReferenceExecutor::create(),
                                        nElems(), &col_idxs_[0]),
                        idx_array::view(gko::ReferenceExecutor::create(),
                                        nElems(), &row_idxs_[0])));

        // if updating system matrix is not needed store ptr in obj registry
        if (!get_update_sys_matrix()) {
            Info << "store sys matrix" << endl;
            const fileName path = sys_matrix_name;
            gkomatrix_ptr = new GKOCOOIOPtr(IOobject(path, db), gkomatrix);
        }
    };

    void set_device_executor(std::shared_ptr<gko::Executor> &device_exec) const
    {
        const word device_executor_name{get_device_executor_name()};
        if (device_executor_name == "omp") {
            device_exec = omp_exec();
        }
        if (device_executor_name == "cuda") {
            device_exec = cuda_exec();
        }
        if (device_executor_name == "hip") {
            device_exec = hip_exec();
        }
        if (device_executor_name == "reference") {
            device_exec = ref_exec();
        }
    };

    void init_sorting_idxs()
    {
        // initialize original index locations
        sorting_idxs_->resize(nElems());

        // set sorting idxs from 0 .. n
        for (int i = 0; i < nElems(); i++) {
            sorting_idxs_->operator[](i) = i;
        }
    };

    void compute_sorting_idxs()
    {
        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        std::stable_sort(sorting_idxs_->data(),
                         &sorting_idxs_->data()[nElems()],
                         [this](size_t i1, size_t i2) {
                             return row_idxs_[i1] < row_idxs_[i2];
                         });
    };

    const std::vector<scalar> &vals() const { return values_; };

    const std::vector<label> &col_idx() const { return col_idxs_; };

    const std::vector<label> &row_idx() const { return row_idxs_; };

    void update_GKOMatrix() const
    {
        // reset vectors
        values_.resize(0);
        col_idxs_.resize(0);
        row_idxs_.resize(0);

        values_.reserve(nElems());
        col_idxs_.reserve(nElems());
        row_idxs_.reserve(nElems());

        // fill vectors unsorted
        for (IndexType i = 0; i < nNeighbours(); ++i) {
            values_.push_back(matrix().lower()[i]);
            row_idxs_.push_back(matrix().lduAddr().lowerAddr()[i]);
            col_idxs_.push_back(matrix().lduAddr().upperAddr()[i]);
        }

        for (IndexType i = 0; i < nCells(); ++i) {
            values_.push_back(matrix().diag()[i]);
            col_idxs_.push_back(i);
            row_idxs_.push_back(i);
        }

        for (IndexType i = 0; i < nNeighbours(); ++i) {
            values_.push_back(matrix().upper()[i]);
            row_idxs_.push_back(matrix().lduAddr().upperAddr()[i]);
            col_idxs_.push_back(matrix().lduAddr().lowerAddr()[i]);
        }
    };

    void sort_GKOMatrix() const
    {
        std::vector<scalar> tmp_values(nElems());
        std::vector<scalar> tmp_col_idxs(nElems());
        std::vector<scalar> tmp_row_idxs(nElems());

        for (label i = 0; i < nElems(); i++) tmp_values[i] = values_[i];
        for (label i = 0; i < nElems(); i++) tmp_col_idxs[i] = col_idxs_[i];
        for (label i = 0; i < nElems(); i++) tmp_row_idxs[i] = row_idxs_[i];

        // TODO parallelise
        for (label i = 0; i < nElems(); i++) {
            label j = sorting_idxs_->operator[](i);
            values_[i] = tmp_values[j];
            col_idxs_[i] = tmp_col_idxs[j];
            row_idxs_[i] = tmp_row_idxs[j];
            // std::cout << "elem " << i << ": " << row_idxs_[i] << " " <<
            // col_idxs_[i] << std::endl;
        }
    };

    scalar compute_residual(mtx *const A, vec *const x, vec *const b) const
    {
        auto one = gko::initialize<vec>({1.0}, ref_exec());
        auto neg_one = gko::initialize<vec>({-1.0}, ref_exec());
        auto res = gko::initialize<vec>({0.0}, ref_exec());
        A->apply(lend(one), lend(x), lend(neg_one), lend(b));
        b->compute_norm2(lend(res));

        return res->at(0);
    };

    virtual solverPerformance solve_impl(word typeName, scalarField &psi,
                                         const scalarField &source,
                                         const direction cmpt = 0) const
    {
        std::shared_ptr<gko::Executor> device_exec{};
        set_device_executor(device_exec);

        // --- Setup class containing solver performance data
        solverPerformance solverPerf(
            lduMatrix::preconditioner::getName(controlDict_) + typeName,
            fieldName_);
        scalarField pA(nCells());
        scalarField wA(nCells());
        matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
        scalar norm_factor = this->normFactor(psi, source, wA, pA);

        auto psi_view = val_array::view(ref_exec(), nCells(),
                                        const_cast<scalar *>(&psi[0]));
        auto source_view = val_array::view(ref_exec(), nCells(),
                                           const_cast<scalar *>(&source[0]));
        auto b =
            vec::create(ref_exec(), gko::dim<2>(nCells(), 1), source_view, 1);

        auto x = vec::create(ref_exec(), gko::dim<2>(nCells(), 1), psi_view, 1);

        std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
            criterion_vec{};

        criterion_vec.push_back(
            OpenFOAMStoppingCriterion::build()
                .with_openfoam_absolute_tolerance(tolerance())
                .with_openfoam_relative_tolerance(0)
                .with_openfoam_norm_factor(norm_factor)
                .with_openfoam_minIter(minIter())
                .with_openfoam_maxIter(maxIter())
                .on(device_exec));

        // Generate solver
        auto solver_gen = this->create_solver(device_exec, criterion_vec);

        // Instantiate a ResidualLogger logger.
        auto logger = std::make_shared<IterationLogger>(device_exec);

        // Add the previously created logger to the solver factory. The
        // logger will be automatically propagated to all solvers created
        // from this factory.
        solver_gen->add_logger(logger);

        auto b_clone = gko::clone(b);
        std::shared_ptr<mtx> gkomatrix{};
        if (gkomatrix_ptr == NULL) {
            gkomatrix = gko::share(
                mtx::create(device_exec, gko::dim<2>(nCells(), nCells()),
                            val_array::view(gko::ReferenceExecutor::create(),
                                            nElems(), &values_[0]),
                            idx_array::view(gko::ReferenceExecutor::create(),
                                            nElems(), &col_idxs_[0]),
                            idx_array::view(gko::ReferenceExecutor::create(),
                                            nElems(), &row_idxs_[0])));
        } else {
            gkomatrix = gkomatrix_ptr->get_ptr();
        }

        if (get_export())
          export_system(fieldName(), gko::lend(gkomatrix), gko::lend(x),
          gko::lend(b));


        solverPerf.initialResidual() = compute_residual(
            gko::lend(gkomatrix), gko::lend(x), gko::lend(b_clone));

        auto solver = solver_gen->generate(gko::share(gkomatrix));

        // Solve system
        SIMPLE_TIME(solve, solver->apply(gko::lend(b), gko::lend(x));)

        auto x_view = val_array::view(ref_exec(), nCells(), x->get_values());
        auto x_clone =
            vec::create(ref_exec(), gko::dim<2>(nCells(), 1), x_view, 1);
        psi_view = x_view;
        b_clone->copy_from(gko::lend(b));

        // solverPerf.finalResidual() =
        //     compute_residual(gko::lend(ginkgo_host_matrix),
        //     gko::lend(x_clone),
        //                      gko::lend(b_clone));

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
