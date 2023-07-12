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

// using dist_mtx = gko::experimental::distributed::Matrix<scalar, label,
// label>;

// template <typename mtx>
// void MatrixInitFunctor<mtx>::update(std::shared_ptr<dist_mtx> &matrix) const
// {
//     // auto values_view =
//     //     val_array::view(values_.get_exec_handler().get_device_exec(),
//     //                     values_.get_global_size(), matrix->get_values());
//     // // copy values to device
//     // values_view = *values_.get_array().get();
// }


// MatrixInitFunctor<gko::matrix::Csr<double>>::update(std::shared_ptr<dist_mtx>);

// template <typename mtx>
// std::shared_ptr<dist_mtx> MatrixInitFunctor<mtx>::init() const
// {
//     using coo_mtx = gko::matrix::Coo<scalar>;

//     label nCells = partition_.get_local_host_size();
//     word msg{"init global csr matrix of size " + std::to_string(nCells)};
//     LOG_1(verbose_, msg)

//     auto values = values_.get_array();
//     auto cols = col_idxs_.get_array();
//     auto rows = row_idxs_.get_array();

//     // check if sorted
//     if (false) {
//         bool is_sorted_rows = true;
//         bool is_sorted_cols = true;
//         auto rows_data = rows->get_const_data();
//         auto cols_data = cols->get_const_data();
//         for (size_t i = 1; i < cols->get_num_elems(); i++) {
//             if (rows_data[i] < rows_data[i - 1]) {
//                 is_sorted_rows = false;
//                 Info << "rows sorting error element " << i << " row[i] "
//                      << rows_data[i] << " row[i-1] " << rows_data[i - 1]
//                      << endl;
//             }
//             // same row but subsequent column is smaller
//             if (cols_data[i] < cols_data[i - 1] &&
//                 rows_data[i] == rows_data[i - 1]) {
//                 is_sorted_cols = false;
//                 Info << "cols sorting error element " << i << " row[i] "
//                      << rows_data[i] << " row[i-1] " << rows_data[i - 1]
//                      << " col[i] " << cols_data[i] << " col[i-1] "
//                      << cols_data[i - 1] << endl;
//             }
//         }

//         Info << "is_sorted rows " << is_sorted_rows << endl;
//         Info << "is_sorted cols " << is_sorted_cols << endl;

//         if (!is_sorted_cols || !is_sorted_rows) {
//             for (size_t i = 1; i < cols->get_num_elems(); i++) {
//                 Info << i << "sparsity (" << rows_data[i] << "," <<
//                 cols_data[i]
//                      << ")\n";
//             }
//         }
//     }
//     auto exec = this->get_exec_handler().get_reference_exec();

//     auto local_mtx =
//         gko::share(coo_mtx::create(exec, gko::dim<2>(nCells, nCells),
//                                    *values.get(), *cols.get(), *rows.get()));


//     auto local_size = partition_.get_local_host_size();

//     gko::matrix_data<scalar, label> A_data;
//     local_mtx->write(A_data);

//     auto comm = this->get_exec_handler().get_gko_mpi_comm_wrapper();
//     auto dist_A = gko::share(dist_mtx::create(*comm.get()));
//     dist_A->read_experimental::distributed(A_data,
//     partition_.get_host_partition().get());

//     if (partition_.get_ranks_per_gpu() == 1) {
//         return dist_A;
//     }

//     // TODO test if this needs to be persistent
//     auto repartitioner =
//         gko::share(gko::experimental::distributed::repartitioner<label,
//         label>::create(
//             *comm.get(), partition_.get_host_partition(),
//             partition_.get_device_partition()));
//     auto to_mat =
//         gko::share(dist_mtx::create(repartitioner->get_to_communicator()));

//     repartitioner->gather(dist_A.get(), to_mat.get());

//     return to_mat;
// }

// MatrixInitFunctor<gko::matrix::Csr<double>>::init();

}  // namespace Foam
