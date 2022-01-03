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
#include <ginkgo/ginkgo.hpp>
#include "CsrMatrixWrapper.H"
#include "common.H"

namespace Foam {

void CsrInitFunctor::update(
    std::shared_ptr<gko::matrix::Csr<scalar>> &csr_matrix) const
{
    if (Pstream::parRun()) {
        if (Pstream::master()) {
            word msg{"update global csr matrix "};
            LOG_1(verbose_, msg)
            auto values_view = val_array::view(
                values_.get_exec_handler().get_device_exec(),
                values_.get_global_size(), csr_matrix->get_values());
            values_view = *values_.get_array().get();
        }
    } else {
        auto values_view = val_array::view(
            values_.get_exec_handler().get_device_exec(),
            values_.get_global_size(), csr_matrix->get_values());
        // copy values to device
        values_view = *values_.get_array().get();
    }
}

std::shared_ptr<gko::matrix::Csr<scalar>> CsrInitFunctor::init() const
{
    using coo_mtx = gko::matrix::Coo<scalar>;

    if (Pstream::parRun()) {
        const auto device_exec = values_.get_exec_handler().get_device_exec();
        const label nCells = global_index_.size();
        word msg{"init global csr matrix of size " + std::to_string(nCells)};
        LOG_1(verbose_, msg)

        auto values = values_.get_global_array();
        auto cols = col_idxs_.get_global_array();
        auto rows = row_idxs_.get_global_array();

        if (Pstream::master()) {
            // for (int i = 0; i < values->get_num_elems(); i++) {
            //     std::cout << "(" << rows->get_const_data()[i] << ","
            //               << cols->get_const_data()[i] << ") "
            //               << values->get_const_data()[i] << std::endl;
            // }
            auto coo_mtx = gko::share(
                coo_mtx::create(device_exec, gko::dim<2>(nCells, nCells),
                                val_array(device_exec, *values.get()),
                                *cols.get(), *rows.get()));

            auto gkomatrix = gko::share(
                mtx::create(device_exec, gko::dim<2>(nCells, nCells)));
            SIMPLE_TIME(verbose_, convert_coo_to_csr,
                        coo_mtx->convert_to(gkomatrix.get());)
            return gkomatrix;
        } else {
            return {};
        }
    } else {
        const auto device_exec = values_.get_exec_handler().get_device_exec();
        const label nCells = global_index_.size();
        word msg{"init csr matrix of size " + std::to_string(nCells)};
        LOG_1(verbose_, msg)
        auto values = values_.get_array();
        auto cols = col_idxs_.get_array();
        auto rows = row_idxs_.get_array();
        for (int i = 0; i < values->get_num_elems(); i++) {
            std::cout << "(" << rows->get_const_data()[i] << ","
                      << cols->get_const_data()[i] << ") "
                      << values->get_const_data()[i] << std::endl;
        }

        auto coo_mtx = gko::share(coo_mtx::create(
            device_exec, gko::dim<2>(nCells, nCells),
            val_array(device_exec, *values_.get_array().get()),
            *col_idxs_.get_array().get(), *row_idxs_.get_array().get()));

        auto gkomatrix =
            gko::share(mtx::create(device_exec, gko::dim<2>(nCells, nCells)));

        SIMPLE_TIME(verbose_, convert_coo_to_csr,
                    coo_mtx->convert_to(gkomatrix.get());)
        return gkomatrix;
    }
}


// void CsrMatrixWrapper::copy_result_back(const scalarField &psi,
//                                           const label nCells) const
// {
//     // TODO rename to host_x
//     auto device_x = vec::create(ref_exec(), gko::dim<2>(nCells, 1));

//     std::vector<std::shared_ptr<vec>> device_xs_ptr{};
//     get_initial_guess(device_xs_ptr);
//     device_x->copy_from(gko::lend(device_xs_ptr[0]));

//     auto x_view = val_array::view(ref_exec(), nCells,
//     device_x->get_values());
//     // for (label i = 0; i < nCells; i++) {
//     //     std::cout << x_view[i] << std::endl;
//     // }

//     // move frome device
//     auto psi_view =
//         val_array::view(ref_exec(), nCells, const_cast<scalar
//         *>(&psi[0]));

//     psi_view = x_view;
// }

// defineTemplateTypeNameWithName(GKOIDXIOPtr, "IDXIOPtr");
// defineTemplateTypeNameWithName(GKOCSRIOPtr, "CSRIOPtr");
// defineTemplateTypeNameWithName(GKOVECIOPtr, "VECIOPtr");

}  // namespace Foam
