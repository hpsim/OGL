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


#include "DevicePersistent/Array/Array.H"
#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"
#include "DevicePersistent/Partition/Partition.H"

#include "fvCFD.H"

namespace Foam {


struct MatrixInitFunctor {
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;

    using val_array = gko::array<scalar>;

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

    const bool regenerate_;

    const label verbose_;

    const word field_name_;

    MatrixInitFunctor(const objectRegistry &db, const ExecutorHandler &exec,
                      const PersistentPartition &partition,
                      const PersistentArray<label> &col_idxs,
                      const PersistentArray<label> &row_idxs,
                      const PersistentArray<scalar> &coeffs,
                      const PersistentArray<label> &non_local_col_idxs,
                      const PersistentArray<label> &non_local_row_idxs,
                      const PersistentArray<scalar> &non_local_coeffs,
                      const word matrix_format, const bool regenerate,
                      const label verbose, const word field_name)
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
          regenerate_(regenerate),
          verbose_(verbose),
          field_name_(field_name)
    {}

    void update(std::shared_ptr<dist_mtx> persistent_device_matrix) const
    {
        if (regenerate_) {
            TIME_WITH_FIELDNAME(verbose_, regenerate_distributed_matrix,
                                field_name_, auto reinit = init();)
            persistent_device_matrix->copy_from(reinit.get());
            return;
        }
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

    std::shared_ptr<dist_mtx> generate_dist_mtx_with_inner_type(
        std::shared_ptr<gko::Executor> exec,
        std::shared_ptr<gko::experimental::mpi::communicator> comm) const
    {
        if (matrix_format_ == "Csr") {
            return dist_mtx::create(exec, *comm.get(),
                                    gko::with_matrix_type<gko::matrix::Csr>());
        }
        if (matrix_format_ == "Ell") {
            return dist_mtx::create(exec, *comm.get(),
                                    gko::with_matrix_type<gko::matrix::Ell>());
        }
        // if (matrix_format_ == "Hybrid") {
        //     return dist_mtx::create(
        //         *comm.get(), gko::with_matrix_type<gko::matrix::Hybrid>());
        // }
        if (matrix_format_ == "Coo") {
            return dist_mtx::create(exec, *comm.get(),
                                    gko::with_matrix_type<gko::matrix::Coo>());
        }

        FatalErrorInFunction << "Matrix format " << matrix_format_
                             << " not supported " << abort(FatalError);
    }

    std::shared_ptr<dist_mtx> init() const
    {
        const label local_size = partition_.get_local_size();
        word msg{"init global csr matrix of size " +
                 std::to_string(local_size)};
        LOG_1(verbose_, msg)

        auto coeffs = coeffs_.get_array();
        auto cols = col_idxs_.get_array();
        auto rows = row_idxs_.get_array();

        auto non_local_coeffs = non_local_coeffs_.get_array();
        auto non_local_cols = non_local_col_idxs_.get_array();
        auto non_local_rows = non_local_row_idxs_.get_array();

        auto exec = exec_.get_ref_exec();

        auto num_rows = partition_.get_local_size();
        gko::device_matrix_data<scalar, label> A_data(
            exec, gko::dim<2>(num_rows, num_rows), *rows.get(), *cols.get(),
            *coeffs.get());

        gko::device_matrix_data<scalar, label> non_local_A_data(
            exec, gko::dim<2>(num_rows, non_local_cols->get_num_elems()),
            *non_local_rows.get(), *non_local_cols.get(),
            *non_local_coeffs.get());


        auto comm = exec_.get_gko_mpi_host_comm();
        auto dist_A = generate_dist_mtx_with_inner_type(
            exec_.get_ref_exec(), exec_.get_gko_mpi_host_comm());

        auto sparse_comm =
            gko::experimental::distributed::sparse_communicator::create(
                *comm.get(), partition_.get_localized_partition());

        auto recv_sizes = sparse_comm->get_recv_sizes();
        auto recv_offs = sparse_comm->get_recv_offsets();
        auto send_sizes = sparse_comm->get_send_sizes();
        auto send_offs = sparse_comm->get_send_offsets();

        dist_A->read_distributed(A_data, non_local_A_data, sparse_comm);

        auto device_mat = generate_dist_mtx_with_inner_type(
            exec_.get_device_exec(), exec_.get_gko_mpi_device_comm());
        dist_A->move_to(device_mat.get());
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
                  controlDict.lookupOrDefault<Switch>("regenerate", false),
                  verbose_, sys_matrix_name),
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
