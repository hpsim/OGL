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
#ifndef OGL_IOGKOMatrixHandler_INCLUDED_H
#define OGL_IOGKOMatrixHandler_INCLUDED_H

#include "fvCFD.H"

#include "../DevicePersistentArray/DevicePersistentArray.H"
#include "../DevicePersistentVector/DevicePersistentVector.H"
#include "../ExecutorHandler/ExecutorHandler.H"
#include "../IOGlobalIndex/gkoGlobalIndex.H"

namespace Foam {


struct MatrixInitFunctor {
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;

    const objectRegistry &db_;

    const ExecutorHandler &exec_;

    const PersistentPartition &partition_;

    const PersistentArray<label> &col_idxs_;

    const PersistentArray<label> &row_idxs_;

    const PersistentArray<scalar> &coeffs_;

    const PersistentArray<label> &non_local_col_idxs_;

    const PersistentArray<label> &non_local_row_idxs_;

    const PersistentArray<scalar> &non_local_coeffs_;

    const word matrix_format_;

    const label verbose_;

    MatrixInitFunctor(const objectRegistry &db, const ExecutorHandler &exec,
                      const PersistentPartition &partition,
                      const PersistentArray<label> &col_idxs,
                      const PersistentArray<label> &row_idxs,
                      const PersistentArray<scalar> &coeffs,
                      const PersistentArray<label> &non_local_col_idxs,
                      const PersistentArray<label> &non_local_row_idxs,
                      const PersistentArray<scalar> &non_local_coeffs,
                      const word matrix_format, const label verbose)
        : db_(db),
          exec_(exec),
          partition_(partition),
          col_idxs_(col_idxs),
          row_idxs_(row_idxs),
          coeffs_(coeffs),
          non_local_col_idxs_(non_local_col_idxs),
          non_local_row_idxs_(non_local_row_idxs),
          non_local_coeffs_(non_local_coeffs),
          matrix_format_(matrix_format),
          verbose_(verbose)
    {}

    void update(std::shared_ptr<dist_mtx> persistent_device_matrix) const
    {
        auto coeffs = coeffs_.get_array();
        auto non_local_coeffs = non_local_coeffs_.get_array();

        auto exec = exec_.get_ref_exec();
        auto device_exec = exec_.get_device_exec();

        scalar *value_ptr{};
        scalar *non_local_value_ptr{};

        std::shared_ptr<const gko::LinOp> local_matrix =
            persistent_device_matrix->get_local_matrix();
        std::shared_ptr<const gko::LinOp> non_local_matrix =
            persistent_device_matrix->get_non_local_matrix();
        if (matrix_format_ == "Csr") {
            gko::matrix::Csr<scalar, label> *coo_ptr =
                (gko::matrix::Csr<scalar, label> *)local_matrix.get();
            value_ptr = coo_ptr->get_values();

            gko::matrix::Csr<scalar, label> *non_local_coo_ptr =
                (gko::matrix::Csr<scalar, label> *)non_local_matrix.get();
            non_local_value_ptr = non_local_coo_ptr->get_values();
        }
        if (matrix_format_ == "Ell") {
            gko::matrix::Ell<scalar, label> *coo_ptr =
                (gko::matrix::Ell<scalar, label> *)local_matrix.get();
            value_ptr = coo_ptr->get_values();

            gko::matrix::Ell<scalar, label> *non_local_coo_ptr =
                (gko::matrix::Ell<scalar, label> *)non_local_matrix.get();
            non_local_value_ptr = non_local_coo_ptr->get_values();
        }
        if (matrix_format_ == "Coo") {
            gko::matrix::Coo<scalar, label> *coo_ptr =
                (gko::matrix::Coo<scalar, label> *)local_matrix.get();
            value_ptr = coo_ptr->get_values();

            gko::matrix::Coo<scalar, label> *non_local_coo_ptr =
                (gko::matrix::Coo<scalar, label> *)non_local_matrix.get();
            non_local_value_ptr = non_local_coo_ptr->get_values();
        }

        if (partition_.get_ranks_per_gpu() != 1) {
            // TODO add field name
            PersistentArray<label> local_scatter_map_pers{
                "local_scatter_map",  //
                db_,
                exec_,
                0,
                verbose_,
                false,
                false};

            // TODO add field name
            PersistentArray<label> non_local_scatter_map_pers{
                "non_local_scatter_map",  //
                db_,
                exec_,
                0,
                verbose_,
                false,
                false};

            auto comm = exec_.get_gko_mpi_host_comm();
            auto repartitioner = gko::share(
                gko::experimental::distributed::repartitioner<
                    label, label>::create(*comm.get(),
                                          partition_.get_host_partition(),
                                          partition_.get_device_partition()));

            auto local_nnz =
                local_scatter_map_pers.get_array()->get_num_elems();
            auto non_local_nnz =
                non_local_scatter_map_pers.get_array()->get_num_elems();

            gko::array<scalar> local_values{exec, local_nnz};
            gko::array<scalar> non_local_values{exec, non_local_nnz};

            repartitioner->update_existing(
                *row_idxs_.get_array(), *non_local_row_idxs_.get_array(),
                *coeffs.get(), *non_local_coeffs.get(),
                *local_scatter_map_pers.get_array().get(),
                *non_local_scatter_map_pers.get_array().get(), local_values,
                non_local_values);

            auto value_view =
                val_array::view(device_exec, local_nnz, value_ptr);

            auto non_local_view = val_array::view(device_exec, non_local_nnz,
                                                  non_local_value_ptr);

            local_values.set_executor(device_exec);
            value_view = local_values;

            non_local_values.set_executor(device_exec);
            non_local_view = non_local_values;
        } else {
            gko::array<scalar> device_values{device_exec, *coeffs.get()};
            gko::array<scalar> device_non_local_values{device_exec,
                                                       *non_local_coeffs.get()};

            auto value_view = val_array::view(
                device_exec, device_values.get_num_elems(), value_ptr);

            auto non_local_view = val_array::view(
                device_exec, device_non_local_values.get_num_elems(),
                non_local_value_ptr);

            value_view = device_values;
            non_local_view = device_non_local_values;
        }
    }

    std::shared_ptr<dist_mtx> generate_dist_mtx_with_inner_type(
        std::shared_ptr<gko::experimental::mpi::communicator> comm) const
    {
        if (matrix_format_ == "Csr") {
            return dist_mtx::create(*comm.get(),
                                    gko::with_matrix_type<gko::matrix::Csr>());
        }
        if (matrix_format_ == "Ell") {
            return dist_mtx::create(*comm.get(),
                                    gko::with_matrix_type<gko::matrix::Ell>());
        }
        // if (matrix_format_ == "Hybrid") {
        //     return dist_mtx::create(
        //         *comm.get(), gko::with_matrix_type<gko::matrix::Hybrid>());
        // }
        if (matrix_format_ == "Coo") {
            return dist_mtx::create(*comm.get(),
                                    gko::with_matrix_type<gko::matrix::Coo>());
        }

        FatalErrorInFunction << "Matrix format " << matrix_format_
                             << " not supported " << abort(FatalError);
    }

    std::shared_ptr<dist_mtx> init() const
    {
        using coo_mtx = gko::matrix::Coo<scalar>;

        label nCells = partition_.get_local_host_size();
        word msg{"init global csr matrix of size " + std::to_string(nCells)};
        LOG_1(verbose_, msg)

        auto coeffs = coeffs_.get_array();
        auto cols = col_idxs_.get_array();
        auto rows = row_idxs_.get_array();

        auto non_local_coeffs = non_local_coeffs_.get_array();
        auto non_local_cols = non_local_col_idxs_.get_array();
        auto non_local_rows = non_local_row_idxs_.get_array();

        // check if sorted
        if (false) {
            bool is_sorted_rows = true;
            bool is_sorted_cols = true;
            auto rows_data = rows->get_const_data();
            auto cols_data = cols->get_const_data();
            for (size_t i = 1; i < cols->get_num_elems(); i++) {
                if (rows_data[i] < rows_data[i - 1]) {
                    is_sorted_rows = false;
                    Info << "rows sorting error element " << i << " row[i] "
                         << rows_data[i] << " row[i-1] " << rows_data[i - 1]
                         << endl;
                }
                // same row but subsequent column is smaller
                if (cols_data[i] < cols_data[i - 1] &&
                    rows_data[i] == rows_data[i - 1]) {
                    is_sorted_cols = false;
                    Info << "cols sorting error element " << i << " row[i] "
                         << rows_data[i] << " row[i-1] " << rows_data[i - 1]
                         << " col[i] " << cols_data[i] << " col[i-1] "
                         << cols_data[i - 1] << endl;
                }
            }

            Info << "is_sorted rows " << is_sorted_rows << endl;
            Info << "is_sorted cols " << is_sorted_cols << endl;

            if (!is_sorted_cols || !is_sorted_rows) {
                for (size_t i = 1; i < cols->get_num_elems(); i++) {
                    Info << i << "sparsity (" << rows_data[i] << ","
                         << cols_data[i] << ")\n";
                }
            }
        }
        auto exec = exec_.get_ref_exec();

        auto num_rows = partition_.get_total_size();
        gko::device_matrix_data<scalar, label> A_data(
            exec, gko::dim<2>(num_rows, num_rows), *rows.get(), *cols.get(),
            *coeffs.get());

        gko::device_matrix_data<scalar, label> non_local_A_data(
            exec, gko::dim<2>(num_rows, num_rows), *non_local_rows.get(),
            *non_local_cols.get(), *non_local_coeffs.get());


        auto comm = exec_.get_gko_mpi_host_comm();
        auto dist_A =
            generate_dist_mtx_with_inner_type(exec_.get_gko_mpi_host_comm());
        dist_A->read_experimental::distributed(
            A_data, non_local_A_data, partition_.get_host_partition().get());

        if (partition_.get_ranks_per_gpu() == 1) {
            auto device_mat = generate_dist_mtx_with_inner_type(
                exec_.get_gko_mpi_device_comm());
            dist_A->move_to(device_mat.get());
            return device_mat;
        }

        // TODO test if this needs to be persistent
        auto repartitioner = gko::share(
            gko::experimental::distributed::repartitioner<label, label>::create(
                *comm.get(), partition_.get_host_partition(),
                partition_.get_device_partition()));
        auto to_mat =
            gko::share(dist_mtx::create(repartitioner->get_to_communicator()));

        auto [local_scatter_map, non_local_scatter_map] =
            repartitioner->gather(dist_A.get(), to_mat.get());

        // TODO add field name
        PersistentArray<label> local_scatter_map_pers{
            local_scatter_map.get_data(),
            "local_scatter_map",
            db_,
            exec_,
            local_scatter_map.get_num_elems(),
            verbose_,
            false,
            false};

        // TODO add field name
        PersistentArray<label> non_local_scatter_map_pers{
            non_local_scatter_map.get_data(),
            "non_local_scatter_map",
            db_,
            exec_,
            non_local_scatter_map.get_num_elems(),
            verbose_,
            false,
            false};

        auto device_mat =
            generate_dist_mtx_with_inner_type(exec_.get_gko_mpi_device_comm());

        to_mat->move_to(device_mat.get());

        return device_mat;
    }
};


class MatrixWrapper {
private:
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;


    const label verbose_;

    const bool export_;

    mutable PersistentBase<dist_mtx, MatrixInitFunctor> gkomatrix_;

    mutable label prev_solve_iters_ = 0;


public:
    MatrixWrapper(const objectRegistry &db, const PersistentExecutor &exec,
                  const PersistentArray<label> &col_idxs,
                  const PersistentArray<label> &row_idxs,
                  const PersistentArray<scalar> &coeffs,
                  const PersistentArray<label> &non_local_col_idxs,
                  const PersistentArray<label> &non_local_row_idxs,
                  const PersistentArray<scalar> &non_local_coeffs,
                  const PersistentPartition &partition,
                  const dictionary &controlDict, const word sys_matrix_name,
                  const label verbose)
        : verbose_(verbose),
          export_(controlDict.lookupOrDefault<Switch>("export", false)),
          gkomatrix_{
              sys_matrix_name + "_matrix", db,
              MatrixInitFunctor(
                  db, exec, partition, col_idxs, row_idxs, coeffs,
                  non_local_col_idxs, non_local_row_idxs, non_local_coeffs,
                  controlDict.lookupOrDefault<word>("matrixFormat", "Coo"),
                  verbose_),
              controlDict.lookupOrDefault<Switch>("updateSysMatrix", true),
              verbose_}
    {}

    std::shared_ptr<gko::LinOp> get() const
    {
        return gkomatrix_.get_persistent_object();
    }


    bool get_export() const { return export_; }
};

}  // namespace Foam

#endif
