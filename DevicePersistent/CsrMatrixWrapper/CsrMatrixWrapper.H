// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once


#include "DevicePersistent/Array/Array.H"
#include "DevicePersistent/ExecutorHandler/ExecutorHandler.H"
#include "DevicePersistent/Partition/Partition.H"

#include "MatrixWrapper/GkoCombinationMatrix/GkoCombinationMatrix.H"
#include "MatrixWrapper/RepartDistMatrix/RepartDistMatrix.H"

#include "fvCFD.H"

namespace Foam {

template <typename InnerMatrixType>
void perform_update(
    std::shared_ptr<const gko::Executor> exec,
    CombinationMatrix<scalar, label, InnerMatrixType> *combination_matrix,
    const PersistentSparsityPattern &sparsity,
    std::shared_ptr<const gko::array<scalar>> coeffs)
{
    int interface_ctr{0};
    for (auto interface :
         combination_matrix->get_combination()->get_operators()) {
        auto span = sparsity.interface_spans_[interface_ctr];
        auto device_value_ptr =
            ((InnerMatrixType *)(interface.get()))->get_values();

        // create a view into data on the device which needs an
        // update
        auto device_value_view =
            val_array::view(exec, span.length(), device_value_ptr);

        // create a view into updated values and copy over to
        // persistent data
        auto update_view = val_array::const_view(
            exec, span.length(), coeffs->get_const_data() + span.begin);

        interface_ctr++;
        device_value_view = update_view;
    }
}


struct MatrixInitFunctor {
    using dist_mtx = RepartDistMatrix<scalar, label, label>;

    using val_array = gko::array<scalar>;

    const objectRegistry &db_;

    const ExecutorHandler &exec_;

    const PersistentPartition &partition_;

    const PersistentSparsityPattern &local_sparsity_;

    const PersistentArray<scalar> &coeffs_;

    const PersistentSparsityPattern &non_local_sparsity_;

    const PersistentArray<scalar> &non_local_coeffs_;

    const word matrix_format_;

    const bool regenerate_;

    const label verbose_;

    const word field_name_;

    MatrixInitFunctor(const objectRegistry &db, const ExecutorHandler &exec,
                      const PersistentPartition &partition,
                      const PersistentSparsityPattern &local_sparsity,
                      const PersistentArray<scalar> &coeffs,
                      const PersistentSparsityPattern &non_local_sparsity,
                      const PersistentArray<scalar> &non_local_coeffs,
                      const word matrix_format, const bool regenerate,
                      const label verbose, const word field_name)
        : db_(db),
          exec_(exec),
          partition_(partition),
          local_sparsity_(local_sparsity),
          coeffs_(coeffs),
          non_local_sparsity_(non_local_sparsity),
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

        // The local and non-local matrix are a CombinationMatrix and we need to
        // update the interfaces iteratively
        std::shared_ptr<const gko::LinOp> local_combination_matrix{
            persistent_device_matrix->get_local_matrix()};
        std::shared_ptr<const gko::LinOp> non_local_combination_matrix{
            persistent_device_matrix->get_non_local_matrix()};
        if (matrix_format_ == "Coo") {
            using Coo = gko::matrix::Coo<scalar, label>;
            perform_update<Coo>(exec_.get_device_exec(),
                                (CombinationMatrix<scalar, label, Coo>
                                     *)(local_combination_matrix.get()),
                                local_sparsity_, coeffs_.get_array());
            perform_update<Coo>(exec_.get_device_exec(),
                                (CombinationMatrix<scalar, label, Coo>
                                     *)(non_local_combination_matrix.get()),
                                non_local_sparsity_,
                                non_local_coeffs_.get_array());
        }
        if (matrix_format_ == "Csr") {
            using Csr = gko::matrix::Csr<scalar, label>;
            perform_update<Csr>(exec_.get_device_exec(),
                                (CombinationMatrix<scalar, label, Csr>
                                     *)(local_combination_matrix.get()),
                                local_sparsity_, coeffs_.get_array());
            perform_update<Csr>(exec_.get_device_exec(),
                                (CombinationMatrix<scalar, label, Csr>
                                     *)(non_local_combination_matrix.get()),
                                non_local_sparsity_,
                                non_local_coeffs_.get_array());
        }
        if (matrix_format_ == "Ell") {
            using Ell = gko::matrix::Ell<scalar, label>;
            perform_update<Ell>(exec_.get_device_exec(),
                                (CombinationMatrix<scalar, label, Ell>
                                     *)(local_combination_matrix.get()),
                                local_sparsity_, coeffs_.get_array());
            perform_update<Ell>(exec_.get_device_exec(),
                                (CombinationMatrix<scalar, label, Ell>
                                     *)(non_local_combination_matrix.get()),
                                non_local_sparsity_,
                                non_local_coeffs_.get_array());
        }
    }

    std::shared_ptr<dist_mtx> generate_dist_mtx_with_inner_type(
        std::shared_ptr<gko::Executor> exec,
        std::shared_ptr<gko::experimental::mpi::communicator> comm) const
    {
        auto local_dim = local_sparsity_.dim_;
        auto non_local_dim = non_local_sparsity_.dim_;
        auto local_interfaces = local_sparsity_.interface_spans_;
        auto non_local_interfaces = non_local_sparsity_.interface_spans_;

        if (matrix_format_ == "Ell") {
            using mtx_type = gko::matrix::Ell<scalar, label>;

            return gko::share(dist_mtx::create(
                exec, comm,
                CombinationMatrix<scalar, label, mtx_type>::create(exec, local_dim,
                                                              local_interfaces),
                CombinationMatrix<scalar, label, mtx_type>::create(
                    exec, non_local_dim, non_local_interfaces)));
        }
        if (matrix_format_ == "Csr") {
            using mtx_type = gko::matrix::Csr<scalar, label>;

            return gko::share(dist_mtx::create(
                exec, comm,
                CombinationMatrix<scalar, label, mtx_type>::create(exec, local_dim,
                                                              local_interfaces),
                CombinationMatrix<scalar, label, mtx_type>::create(
                    exec, non_local_dim, non_local_interfaces)));
        }
        if (matrix_format_ == "Coo") {
            using mtx_type = gko::matrix::Coo<scalar, label>;

            return gko::share(dist_mtx::create(
                exec, comm,
                CombinationMatrix<scalar, label, mtx_type>::create(exec, local_dim,
                                                              local_interfaces),
                CombinationMatrix<scalar, label, mtx_type>::create(
                    exec, non_local_dim, non_local_interfaces)));
        }

        FatalErrorInFunction << "Matrix format " << matrix_format_
                             << " not supported. Supported formats are: Ell, Csr, and Coo."
                             << abort(FatalError);
    }

    std::shared_ptr<dist_mtx> init() const
    {
        const label local_size = partition_.get_local_size();
        word msg{"init global csr matrix of size " +
                 std::to_string(local_size)};
        LOG_1(verbose_, msg)

        auto exec = exec_.get_ref_exec();

        auto coeffs = coeffs_.get_array();
        auto cols = local_sparsity_.col_idxs_.get_array();
        auto rows = local_sparsity_.row_idxs_.get_array();
        gko::device_matrix_data<scalar, label> local_A_data{
            exec, local_sparsity_.dim_, *rows.get(), *cols.get(),
            *coeffs.get()};

        auto non_local_coeffs = non_local_coeffs_.get_array();
        auto non_local_cols = non_local_sparsity_.col_idxs_.get_array();
        auto non_local_rows = non_local_sparsity_.row_idxs_.get_array();
        gko::device_matrix_data<scalar, label> non_local_A_data{
            exec, non_local_sparsity_.dim_, *non_local_rows.get(),
            *non_local_cols.get(), *non_local_coeffs.get()};

        auto comm = exec_.get_gko_mpi_host_comm();

        // TODO here we would need a source and target comm / partition to allow
        // repartitioning
        auto sparse_comm =
            gko::experimental::distributed::sparse_communicator::create(
                *comm.get(), partition_.get_localized_partition());

        auto dist_A = generate_dist_mtx_with_inner_type(
            exec_.get_device_exec(), exec_.get_gko_mpi_host_comm());

        // TODO here we would need a source and target comm / partition to allow
        // repartitioning However repartitioning is also specific to the
        // combination matrix since it moves data between interfaces thus it
        // would need a function that does repartition(local, non_local) we
        // could do that by implementing a wrapper for the distributed matrix
        // type GKORepartitionedDistributedMatrix after the first read it
        // constructs everything needed for repartitioning update would than be
        // handled via repeat_create()
        dist_A->read_distributed(local_A_data, non_local_A_data, sparse_comm);

        // auto device_mat = generate_dist_mtx_with_inner_type(
        //     exec_.get_device_exec(), exec_.get_gko_mpi_device_comm());
        // dist_A->move_to(device_mat.get());
        return dist_A;
    }
};


class MatrixWrapper {
private:
    using dist_mtx = RepartDistMatrix<scalar, label, label>;

    const objectRegistry &db_;

    const label verbose_;

    const bool export_;

    const word field_name_;

    const word matrix_format_;

    mutable PersistentBase<dist_mtx, MatrixInitFunctor> gkomatrix_;

    mutable label prev_solve_iters_ = 0;


public:
    MatrixWrapper(const objectRegistry &db, const PersistentExecutor &exec,
                  const PersistentSparsityPattern &local_sparsity,
                  const PersistentArray<scalar> &coeffs,
                  const PersistentSparsityPattern &non_local_sparsity,
                  const PersistentArray<scalar> &non_local_coeffs,
                  const PersistentPartition &partition,
                  const dictionary &controlDict, const word sys_matrix_name,
                  const label verbose)
        : db_(db),
          verbose_(verbose),
          export_(controlDict.lookupOrDefault<Switch>("export", false)),
          field_name_(sys_matrix_name),
          matrix_format_(
              controlDict.lookupOrDefault<word>("matrixFormat", "Coo")),
          gkomatrix_{
              sys_matrix_name + "_matrix", db,
              MatrixInitFunctor(
                  db, exec, partition, local_sparsity, coeffs,
                  non_local_sparsity, non_local_coeffs, matrix_format_,
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

    /** Exports local and non-local matrix to processor?/<time>/.mtx files
     */
    void write() const
    {
        auto dist_matrix = gko::as<
            gko::experimental::distributed::Matrix<scalar, label, label>>(
            this->get());

        if (matrix_format_ == "Coo") {
            using Coo = gko::matrix::Coo<scalar, label>;
            std::vector<std::shared_ptr<const gko::LinOp>> local_interfaces =
                gko::as<CombinationMatrix<scalar, label, Coo>>(
                    dist_matrix->get_local_matrix())
                    ->get_combination()
                    ->get_operators();
            auto non_local_interfaces =
                gko::as<CombinationMatrix<scalar, label, Coo>>(
                    dist_matrix->get_non_local_matrix())
                    ->get_combination()
                    ->get_operators();
            export_mtx<Coo>(word(field_name_ + "_local"), local_interfaces,
                            db_);
            export_mtx<Coo>(field_name_ + "_non_local", non_local_interfaces,
                            db_);
        }

        if (matrix_format_ == "Csr") {
            using Csr = gko::matrix::Csr<scalar, label>;
            auto local_interfaces =
                gko::as<CombinationMatrix<scalar, label, Csr>>(
                    dist_matrix->get_local_matrix().get())
                    ->get_combination()
                    ->get_operators();
            auto non_local_interfaces =
                gko::as<CombinationMatrix<scalar, label, Csr>>(
                    dist_matrix->get_non_local_matrix().get())
                    ->get_combination()
                    ->get_operators();
            export_mtx<Csr>(word(field_name_ + "_local"), local_interfaces,
                            db_);
            export_mtx<Csr>(field_name_ + "_non_local", non_local_interfaces,
                            db_);
        }

        if (matrix_format_ == "Ell") {
            using Ell = gko::matrix::Ell<scalar, label>;
            auto local_interfaces =
                gko::as<CombinationMatrix<scalar, label, Ell>>(
                    dist_matrix->get_local_matrix().get())
                    ->get_combination()
                    ->get_operators();
            auto non_local_interfaces =
                gko::as<CombinationMatrix<scalar, label, Ell>>(
                    dist_matrix->get_non_local_matrix().get())
                    ->get_combination()
                    ->get_operators();
            export_mtx<Ell>(field_name_ + "_local", local_interfaces, db_);
            export_mtx<Ell>(field_name_ + "_non_local", non_local_interfaces,
                            db_);
        }
    }
};

}  // namespace Foam
