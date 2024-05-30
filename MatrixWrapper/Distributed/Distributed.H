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
    Foam::IOSortingIdxHandler

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    CommunicationPattern.H

\*---------------------------------------------------------------------------*/
#include <map>

#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/ginkgo.hpp>


#include "MatrixWrapper/CommunicationPattern/CommunicationPattern.H"
#include "MatrixWrapper/LDUMatrix/HostMatrix.H"
#include "Repartitioner/Repartitioner.H"


template <typename DistMtxType>
std::unique_ptr<DistMtxType> generate_dist_mtx_with_inner_type(
    word matrix_format, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::experimental::distributed::sparse_communicator>
        comm,
    std::shared_ptr<const SparsityPattern> local_sparsity,
    std::shared_ptr<const SparsityPattern> non_local_sparsity)
{
    auto local_dim = local_sparsity->dim;
    auto non_local_dim = non_local_sparsity->dim;
    auto local_interfaces = local_sparsity->interface_spans;
    auto non_local_interfaces = non_local_sparsity->interface_spans;

    if (matrix_format == "Ell") {
        using mtx_type = gko::matrix::Ell<scalar, label>;

        return DistMtxType::create(
            exec, comm,
            CombinationMatrix<scalar, label, mtx_type>::create(
                exec, local_dim, local_interfaces),
            CombinationMatrix<scalar, label, mtx_type>::create(
                exec, non_local_dim, non_local_interfaces));
    }
    if (matrix_format == "Csr") {
        using mtx_type = gko::matrix::Csr<scalar, label>;

        return DistMtxType::create(
            exec, comm,
            CombinationMatrix<scalar, label, mtx_type>::create(
                exec, local_dim, local_interfaces),
            CombinationMatrix<scalar, label, mtx_type>::create(
                exec, non_local_dim, non_local_interfaces));
    }
    if (matrix_format == "Coo") {
        using mtx_type = gko::matrix::Coo<scalar, label>;

        return DistMtxType::create(
            exec, comm,
            CombinationMatrix<scalar, label, mtx_type>::create(
                exec, local_dim, local_interfaces),
            CombinationMatrix<scalar, label, mtx_type>::create(
                exec, non_local_dim, non_local_interfaces));
    }

    FatalErrorInFunction << "Matrix format " << matrix_format
                         << " not supported! Supported formats: Csr, Ell, Coo"
                         << abort(FatalError);
    return {};
}


// The RepartDistMatrix class is a wrapper around Ginkgos distributed Matrix
// class
//
// It adds functionality for repeated read and repartitioning operatitions. As a
// constraint it is required that the inner matrix types of the distributed
// matrix are of RepartDistMatrix type.
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
class RepartDistMatrix
    : public gko::experimental::EnableDistributedLinOp<
          RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>,
      public gko::EnableCreateMethod<
          RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>,
      public gko::experimental::distributed::DistributedBase {
    friend class gko::EnableCreateMethod<
        RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>;
    friend class gko::EnablePolymorphicObject<
        RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>,
        gko::LinOp>;

    using dist_mtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                               GlobalIndexType>;

    using part_type = gko::experimental::distributed::Partition<label, label>;
    using local_part_type =
        gko::experimental::distributed::localized_partition<label>;

    using vec = gko::matrix::Dense<ValueType>;
    using device_matrix_data = gko::device_matrix_data<scalar, label>;
    using communicator = gko::experimental::mpi::communicator;
    using sparse_communicator =
        gko::experimental::distributed::sparse_communicator;

public:
    using gko::experimental::EnableDistributedLinOp<RepartDistMatrix<
        ValueType, LocalIndexType, GlobalIndexType>>::convert_to;
    using gko::experimental::EnableDistributedLinOp<
        RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>::move_to;

    std::shared_ptr<const gko::LinOp> get_local_matrix()
    {
        this->dist_mtx_->get_local_matrix();
    }

    std::shared_ptr<const gko::LinOp> get_non_local_matrix()
    {
        this->dist_mtx_->get_non_local_matrix();
    }

    /**
     * Copy-assigns a CombinationMatrix matrix. Preserves executor, copies
     * everything else.
     */
    RepartDistMatrix &operator=(const RepartDistMatrix &other)
    {
        FatalErrorInFunction << "Copying the RepartDistMatrix is disallowed "
                                "for performance reasons"
                             << abort(FatalError);
    }

    /**
     * Move-assigns a CombinationMatrix matrix. Preserves executor, moves the
     * data and leaves the moved-from object in an empty state (0x0 LinOp with
     * unchanged executor and strategy, no nonzeros and valid row pointers).
     */
    RepartDistMatrix &operator=(RepartDistMatrix &&other)
    {
        if (&other != this) {
            gko::experimental::EnableDistributedLinOp<
                RepartDistMatrix>::operator=(std::move(other));
            this->dist_mtx_ = std::move(other.dist_mtx_);
            this->set_size(this->dist_mtx_->get_size());
        }
        return *this;
    }


    std::shared_ptr<const part_type> get_partition() const
    {
        return this->partition_;
    }

    static std::shared_ptr<RepartDistMatrix> create(
        word matrix_format, const Repartitioner &repartitioner,
        std::shared_ptr<const HostMatrixWrapper> host_A)
    {
        auto &exec_handler = host_A.get()->get_exec_handler();
        auto exec = exec_handler.get_device_exec();
        auto comm = *exec_handler.get_communicator().get();

        // repartition things here first by creating device_matrix_data on
        // device with correct size on each rank
        //
        // NOTE Start by copying first and repartition later
        auto local_sparsity_ = host_A->compute_local_sparsity(exec);
        auto non_local_sparsity_ = host_A->compute_non_local_sparsity(exec);

        auto orig_partition = repartitioner.get_orig_partition();

        // create original communicator pattern
        auto ranks_per_gpu = repartitioner.get_ranks_per_gpu();
        auto src_comm_pattern =
            host_A.get()->create_communication_pattern(comm);
        // create partiton here and pass to constructor
        auto dst_comm_pattern = repartitioner.repartition_comm_pattern(
            src_comm_pattern, orig_partition);

        label rank{src_comm_pattern.get_comm().rank()};
        label owner_rank = repartitioner.get_owner_rank();
        bool owner = repartitioner.is_owner();

        auto [repart_loc_sparsity, repart_non_loc_sparsity] =
            repartitioner.repartition_sparsity(local_sparsity_,
                                               non_local_sparsity_);

        local_sparsity_ = repart_loc_sparsity;
        non_local_sparsity_ = repart_non_loc_sparsity;

        std::cout << __FILE__ << " rank " << rank
                  << " build_localized_partition "
                  << " dim " << local_sparsity_->dim[0] << " send idxs size "
                  << dst_comm_pattern.send_idxs.size() << " target ids "
                  << dst_comm_pattern.target_ids << " target sizes "
                  << dst_comm_pattern.target_sizes << "\n";

        auto localized_partition = local_part_type::build_from_blocked_recv(
            exec, local_sparsity_->dim[0], dst_comm_pattern.send_idxs,
            dst_comm_pattern.target_ids, dst_comm_pattern.target_sizes);

        std::cout << __FILE__ << " rank " << rank << " local sparsity size "
                  << local_sparsity_->size_ << " local sparsity dim ["
                  << local_sparsity_->dim[0] << "x" << local_sparsity_->dim[1]
                  << "] non_local sparsity size " << non_local_sparsity_->size_
                  << " non local sparsity dim [" << non_local_sparsity_->dim[0]
                  << "x" << non_local_sparsity_->dim[1] << "] target_ids "
                  << dst_comm_pattern.target_ids << " target_sizes "
                  << dst_comm_pattern.target_sizes << " target_send_idxs.size "
                  << dst_comm_pattern.send_idxs.size()
                  << " non_local_sparsity.size " << non_local_sparsity_->size_
                  << " get_recv_indices "
                  << localized_partition->get_recv_indices().get_num_elems()
                  << " \n";

        auto sparse_comm =
            sparse_communicator::create(comm, localized_partition);

        auto dist_A = generate_dist_mtx_with_inner_type<dist_mtx>(
            matrix_format, exec, sparse_comm, local_sparsity_,
            non_local_sparsity_);

        std::cout << __FILE__ << " rank " << rank
                  << " done generate dist_mtx\n";

        auto local_coeffs = gko::array<scalar>(exec, local_sparsity_->size_);
        auto non_local_coeffs =
            gko::array<scalar>(exec, non_local_sparsity_->size_);


        if (owner) {
            local_coeffs.fill(0.0);
            non_local_coeffs.fill(0.0);
        }

        // FIXME make sure that we work on the device executor
        dist_A->read_distributed(
            device_matrix_data(exec, local_sparsity_->dim,
                               local_sparsity_->row_idxs,
                               local_sparsity_->col_idxs, local_coeffs),
            device_matrix_data(exec, non_local_sparsity_->dim,
                               non_local_sparsity_->row_idxs,
                               non_local_sparsity_->col_idxs, non_local_coeffs),
            sparse_comm);


        label nrows = host_A->get_local_nrows();
        label local_matrix_nnz = host_A->get_local_matrix_nnz();
        // size + padding has to be local_matrix_nnz
        auto diag_comm_pattern = compute_send_recv_counts(
            exec_handler, ranks_per_gpu, nrows, local_matrix_nnz,
            local_matrix_nnz - nrows, 0);

        label upper_nnz = host_A->get_upper_nnz();
        auto upper_comm_pattern = compute_send_recv_counts(
            exec_handler, ranks_per_gpu, upper_nnz, local_matrix_nnz, 0,
            local_matrix_nnz - upper_nnz);
        auto lower_comm_pattern =
            compute_send_recv_counts(exec_handler, ranks_per_gpu, upper_nnz,
                                     local_matrix_nnz, upper_nnz, nrows);

        scalar *local_ptr;
        label local_interfaces_count = 0;

        // update values
        if (owner) {
            using Coo = gko::matrix::Coo<scalar, label>;
            std::shared_ptr<const Coo> local =
                gko::as<Coo>(gko::as<CombinationMatrix<scalar, label, Coo>>(
                                 dist_A->get_local_matrix())
                                 ->get_combination()
                                 ->get_operators()[0]);
            local_interfaces_count =
                gko::as<CombinationMatrix<scalar, label, Coo>>(
                    dist_A->get_local_matrix())
                    ->get_combination()
                    ->get_operators()
                    .size();

            local_ptr = const_cast<scalar *>(local->get_const_values());
        }
        communicate_values(exec_handler, diag_comm_pattern, host_A->get_diag(),
                           local_ptr);
        communicate_values(exec_handler, upper_comm_pattern,
                           host_A->get_upper(), local_ptr);
        if (host_A->symmetric()) {
            // TODO FIXME
            // if symmetric we can reuse already copied data
            communicate_values(exec_handler, lower_comm_pattern,
                               host_A->get_lower(), local_ptr);
        } else {
            communicate_values(exec_handler, lower_comm_pattern,
                               host_A->get_lower(), local_ptr);
        }

        if (owner) {
            // next update local and non_local interfaces
            // TODO FIXME
            for (int i = 1; i < local_interfaces_count; i++) {
                using Coo = gko::matrix::Coo<scalar, label>;
                std::shared_ptr<const Coo> local =
                    gko::as<Coo>(gko::as<CombinationMatrix<scalar, label, Coo>>(
                                     dist_A->get_local_matrix())
                                     ->get_combination()
                                     ->get_operators()[i]);
                local_ptr = const_cast<scalar *>(local->get_const_values());

                // for (int j = 0; j < 10; j++) {
                //     local_ptr[j] = 1.0;
                // }
            }
        }

        // reorder updated values
        if (owner) {
            // NOTE local sparsity size includes the interfaces
            using Coo = gko::matrix::Coo<scalar, label>;
            using dim_type = gko::dim<2>::dimension_type;
            std::shared_ptr<const Coo> local =
                gko::as<Coo>(gko::as<CombinationMatrix<scalar, label, Coo>>(
                                 dist_A->get_local_matrix())
                                 ->get_combination()
                                 ->get_operators()[0]);
            auto local_elements = local->get_num_stored_elements();
            local_ptr = const_cast<scalar *>(local->get_const_values());
            // TODO make sure this doesn't copy
            // create a non owning dense matrix of local_values

            auto row_collection = gko::share(gko::matrix::Dense<scalar>::create(
                exec, gko::dim<2>{static_cast<dim_type>(local_elements), 1},
                gko::array<scalar>::view(exec, local_elements, local_ptr), 1));

            auto mapping_view = gko::array<label>::view(
                 exec, local_elements, local_sparsity_->ldu_mapping.get_data());


            // TODO this needs to copy ldu_mapping to the device
            auto dense_vec = row_collection->clone();
            dense_vec->row_gather(&mapping_view,
                                  row_collection.get());
            if (rank == 0) {
                std::cout << __FILE__ << " on rank " << rank
                          << " done row gather\n";
            }
        }

        std::cout << __FILE__ << " on rank " << rank << " make_shared "
                  << " dist_A size " << dist_A->get_size() << "\n";
        auto ret = std::make_shared<RepartDistMatrix>(
            exec, comm, repartitioner.get_repart_dim(), dist_A->get_size(),
            std::move(dist_A));
        if (rank == 0) {
            std::cout << __FILE__ << " on rank " << rank
                      << "done make_shared\n";
        }

        return ret;
    };

    // TODO matrix_format can be data member
    void update(
        word matrix_format, const Repartitioner &repartitioner,
        std::shared_ptr<const HostMatrixWrapper> host_A)
    {
        auto &exec_handler = host_A.get()->get_exec_handler();
        auto device_exec = exec_handler.get_device_exec();

        scalar *local_value_ptr{};
        scalar *non_local_value_ptr{};

        FatalErrorInFunction << "Not implemented" << abort(FatalError);
        // TODO local matrix is now of type Combination<Csr>
        // Combination should get an update(gko::array<> coeffs function)
        //
        // if (matrix_format == "Csr") {
        //     local_value_ptr =
        //         (gko::matrix::Csr<scalar, label>
        //         *)(dist_mtx_->get_local_matrix())
        //             ->get_values();

        //     non_local_value_ptr = (gko::matrix::Csr<scalar, label>
        //     *)dist_mtx_
        //                               ->get_non_local_matrix()
        //                               ->get_values();
        // }
        // if (matrix_format == "Ell") {
        //     local_value_ptr =
        //         (gko::matrix::Ell<scalar, label>
        //         *)dist_mtx_->get_local_matrix()
        //             ->get_values();

        //     non_local_value_ptr = (gko::matrix::Ell<scalar, label>
        //     *)dist_mtx_
        //                               ->get_non_local_matrix()
        //                               ->get_values();
        // }
        // if (matrix_format == "Coo") {
        //     local_value_ptr =
        //         (gko::matrix::Coo<scalar, label>
        //         *)dist_mtx_->get_local_matrix()
        //             ->get_values();

        //     non_local_value_ptr = (gko::matrix::Coo<scalar, label>
        //     *)dist_mtx_
        //                               ->get_non_local_matrix()
        //                               ->get_values();
        // }

        const label local_nnz = local_sparsity_->size_;
        const label non_local_nnz = non_local_sparsity_->size_;
        auto local_coeffs = gko::array<scalar>(device_exec, local_nnz);
        auto non_local_coeffs = gko::array<scalar>(device_exec, non_local_nnz);

        host_A.get()->compute_local_coeffs(local_sparsity_, local_coeffs);
        host_A.get()->compute_non_local_coeffs(non_local_sparsity_,
                                               non_local_coeffs);

        auto value_view =
            val_array::view(device_exec, local_nnz, local_value_ptr);

        auto non_local_view =
            val_array::view(device_exec, non_local_nnz, non_local_value_ptr);

        value_view = local_coeffs;
        non_local_view = non_local_coeffs;

        // On update dist_mtx already exists
        // and only local and non local coeffs need to be updated
        //
        // repartition_coeffs(local_device_data, non_local_device_data)
    };

    RepartDistMatrix(std::shared_ptr<const gko::Executor> exec,
                     communicator comm, gko::dim<2> local_size,
                     gko::dim<2> global_size,
                     std::unique_ptr<dist_mtx> dist_mtx)
        : gko::experimental::EnableDistributedLinOp<RepartDistMatrix>(exec),
          gko::experimental::distributed::DistributedBase(comm),
          dist_mtx_(std::move(dist_mtx))
    {
        this->set_size(global_size);
        // partition_ = gko::share(
        //     gko::experimental::distributed::build_partition_from_local_size<
        //         label, label>(exec, comm, local_size[0]));
    }

    void write(const word field_name_, const objectRegistry &db_) const
    {
        auto dist_matrix = gko::as<
            gko::experimental::distributed::Matrix<scalar, label, label>>(
            this->dist_mtx_.get());
        // TODO FIXME dont leave hardcoded value here
        word matrix_format_{"Coo"};

        if (matrix_format_ == "Coo") {
            using Coo = gko::matrix::Coo<scalar, label>;
            std::vector<std::shared_ptr<const gko::LinOp>> local_interfaces =
                gko::as<CombinationMatrix<scalar, label, Coo>>(
                    dist_matrix->get_local_matrix())
                    ->get_combination()
                    ->get_operators();
            export_mtx<Coo>(word(field_name_ + "_local"), local_interfaces,
                            db_);
            auto non_local_interfaces =
                gko::as<CombinationMatrix<scalar, label, Coo>>(
                    dist_matrix->get_non_local_matrix())
                    ->get_combination()
                    ->get_operators();
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

    RepartDistMatrix(std::shared_ptr<const gko::Executor> exec,
                     communicator comm)
        : gko::experimental::EnableDistributedLinOp<RepartDistMatrix>(exec),
          gko::experimental::distributed::DistributedBase{comm}
    {}

protected:
    // Here we implement the application of the linear operator, x = A * b.
    // apply_impl will be called by the apply method, after the arguments
    // have been moved to the correct executor and the operators checked for
    // conforming sizes.
    //
    // For simplicity, we assume that there is always only one right hand
    // side and the stride of consecutive elements in the vectors is 1 (both
    // of these are always true in this example).
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        std::cout << __FILE__ << __LINE__ << "call apply\n";
        this->dist_mtx_->apply(b, x);
    }


    // There is also a version of the apply function which does the
    // operation x = alpha * A * b + beta * x. This function is commonly
    // used and can often be better optimized than implementing it using x =
    // A * b. However, for simplicity, we will implement it exactly like
    // that in this example.
    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {
        this->dist_mtx_->apply(alpha, b, beta, x);
    }


private:
    std::unique_ptr<dist_mtx> dist_mtx_;

    std::shared_ptr<part_type> partition_;

    std::shared_ptr<SparsityPattern> local_sparsity_;

    std::shared_ptr<SparsityPattern> non_local_sparsity_;
};
