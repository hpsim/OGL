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
#include <ginkgo/ginkgo.hpp>
#include "IOGKOMatrixHandler.H"

namespace Foam {

void IOGKOMatrixHandler::init_device_matrix(
    const objectRegistry &db, std::vector<scalar> &values_host,
    std::vector<label> &col_idxs_host, std::vector<label> &row_idxs_host,
    const label nElems, const label nCells, const bool store) const
{
    std::shared_ptr<gko::Executor> device_exec = get_device_executor();

    if (sys_matrix_stored_) {
        gkomatrix_ptr_ = &db.lookupObjectRef<GKOCSRIOPtr>(sys_matrix_name_);
        return;
    }

    std::shared_ptr<idx_array> col_idx;
    std::shared_ptr<idx_array> row_idx;
    if (sparsity_pattern_stored_) {
        io_col_idxs_ptr_ =
            &db.lookupObjectRef<GKOIDXIOPtr>(sparsity_pattern_name_cols_);
        io_row_idxs_ptr_ =
            &db.lookupObjectRef<GKOIDXIOPtr>(sparsity_pattern_name_rows_);
        col_idx = io_col_idxs_ptr_->get_ptr();
        row_idx = io_row_idxs_ptr_->get_ptr();
    } else {
        // if not stored yet create sparsity pattern from correspondent
        // views
        auto col_idx_view = idx_array::view(gko::ReferenceExecutor::create(),
                                            nElems, &col_idxs_host[0]);
        auto row_idx_view = idx_array::view(gko::ReferenceExecutor::create(),
                                            nElems, &row_idxs_host[0]);

        // copy sparsity pattern to device and leave it there
        col_idx = std::make_shared<idx_array>(col_idx_view);
        col_idx->set_executor(device_exec);
        row_idx = std::make_shared<idx_array>(row_idx_view);
        row_idx->set_executor(device_exec);
    }

    // if system matrix is not stored create it and set shared pointer
    auto coo_mtx = gko::share(
        coo_mtx::create(device_exec, gko::dim<2>(nCells, nCells),
                        val_array::view(gko::ReferenceExecutor::create(),
                                        nElems, &values_host[0]),
                        *col_idx.get(), *row_idx.get()));

    auto gkomatrix =
        gko::share(mtx::create(device_exec, gko::dim<2>(nCells, nCells)));

    coo_mtx->convert_to(gkomatrix.get());


    // if updating system matrix is not needed store ptr in obj registry
    if (store) {
        const fileName path = sys_matrix_name_;
        gkomatrix_ptr_ = new GKOCSRIOPtr(IOobject(path, db), gkomatrix);
    } else {
        gkomatrix_ptr_ = NULL;
        gkomatrix_ = gkomatrix;
    }


    // in any case store sparsity pattern
    const fileName path_col = sparsity_pattern_name_cols_;
    const fileName path_row = sparsity_pattern_name_rows_;
    io_col_idxs_ptr_ = new GKOIDXIOPtr(IOobject(path_col, db), col_idx);
    io_row_idxs_ptr_ = new GKOIDXIOPtr(IOobject(path_row, db), row_idx);
};


defineTemplateTypeNameWithName(GKOIDXIOPtr, "IDXIOPtr");
defineTemplateTypeNameWithName(GKOCSRIOPtr, "CSRIOPtr");
defineTemplateTypeNameWithName(GKOVECIOPtr, "VECIOPtr");

}  // namespace Foam
