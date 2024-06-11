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
    gko::experimental::mpi::communicator comm,
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

    using vec = gko::matrix::Dense<ValueType>;
    using device_matrix_data = gko::device_matrix_data<scalar, label>;
    using communicator = gko::experimental::mpi::communicator;

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
        if (&other != this) {
            // FatalErrorInFunction << "Copying the RepartDistMatrix is
            // disallowed "
            //                         "for performance reasons"
            //                      << abort(FatalError);
            gko::experimental::EnableDistributedLinOp<
                RepartDistMatrix>::operator=(std::move(other));
            this->dist_mtx_ = other.dist_mtx_;
            this->local_sparsity_ = other.local_sparsity_;
            this->non_local_sparsity_ = other.non_local_sparsity_;
            this->src_comm_pattern_ = other.src_comm_pattern_;
            this->local_interfaces_ = other.local_interfaces_;
        }
        return *this;
    }

    /**
     * Move-assigns a CombinationMatrix matrix. Preserves executor, moves the
     * data and leaves the moved-from object in an empty state (0x0 LinOp with
     * unchanged executor and strategy, no nonzeros and valid row pointers).
     */
    RepartDistMatrix &operator=(RepartDistMatrix &&other)
    {
        if (&other != this) {
            FatalErrorInFunction << "Not implemented" << abort(FatalError);
            gko::experimental::EnableDistributedLinOp<
                RepartDistMatrix>::operator=(std::move(other));
            this->dist_mtx_ = std::move(other.dist_mtx_);
            this->local_sparsity_ = std::move(other.local_sparsity_);
            this->non_local_sparsity_ = std::move(other.non_local_sparsity_);
            this->src_comm_pattern_ = std::move(other.src_comm_pattern_);
            this->local_interfaces_ = std::move(other.local_interfaces_);
        }
        return *this;
    }

    static std::shared_ptr<RepartDistMatrix> create(
        const ExecutorHandler &exec_handler, word matrix_format,
        const Repartitioner &repartitioner,
        std::shared_ptr<const HostMatrixWrapper> host_A)
    {
        auto exec = exec_handler.get_ref_exec();
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
        label rank{comm.rank()};

        auto src_comm_pattern =
            host_A.get()->create_communication_pattern(exec_handler);
        // create partiton here and pass to constructor
        //
        auto dst_comm_pattern = repartitioner.repartition_comm_pattern(
            exec_handler, src_comm_pattern, orig_partition);

        label owner_rank = repartitioner.get_owner_rank(exec_handler);
        bool owner = repartitioner.is_owner(exec_handler);

        auto [repart_loc_sparsity, repart_non_loc_sparsity, local_interfaces] =
            repartitioner.repartition_sparsity(exec_handler, local_sparsity_,
                                               non_local_sparsity_);

        local_sparsity_ = repart_loc_sparsity;
        non_local_sparsity_ = repart_non_loc_sparsity;


        auto device_exec = exec_handler.get_device_exec();
        auto dist_A = gko::share(generate_dist_mtx_with_inner_type<dist_mtx>(
            matrix_format, device_exec, comm, local_sparsity_,
            non_local_sparsity_));

        auto local_coeffs = gko::array<scalar>(exec, local_sparsity_->size_);
        auto non_local_coeffs =
            gko::array<scalar>(exec, non_local_sparsity_->size_);


        if (owner) {
            local_coeffs.fill(0.0);
            non_local_coeffs.fill(0.0);
        }

        std::cout << __FILE__ << " rank " << rank << " local sparsity size "
                  << local_sparsity_->size_ << " local sparsity dim ["
                  << local_sparsity_->dim[0] << "x" << local_sparsity_->dim[1]
                  << "] non_local sparsity size " << non_local_sparsity_->size_
                  << " non local sparsity dim [" << non_local_sparsity_->dim[0]
                  << "x" << non_local_sparsity_->dim[1] << "] target_ids "
                  << dst_comm_pattern->target_ids << " target_sizes "
                  << dst_comm_pattern->target_sizes << " target_send_idxs.size "
                  << dst_comm_pattern->send_idxs.size()
                  << " non_local_sparsity.size " << non_local_sparsity_->size_
                  << " get_recv_indices "
                  << localized_partition->get_recv_indices().get_num_elems()
                  << " \n";

        // FIXME make sure that we work on the device executor
        dist_A->read_distributed(
            device_matrix_data(exec, local_sparsity_->dim,
                               local_sparsity_->row_idxs,
                               local_sparsity_->col_idxs, local_coeffs),
            device_matrix_data(
                exec, non_local_sparsity_->dim, non_local_sparsity_->row_idxs,
                non_local_sparsity_->col_idxs, non_local_coeffs));


        // auto dist_A = DistMtxType::create(
        //        exec, comm,
        //        CombinationMatrix<scalar, label, mtx_type>::create(
        //            exec, local_dim, local_interfaces),
        //        CombinationMatrix<scalar, label, mtx_type>::create(
        //            exec, non_local_dim, non_local_interfaces));

        update_impl(exec_handler, matrix_format, repartitioner, host_A, dist_A,
                    local_sparsity_, non_local_sparsity_, src_comm_pattern,
                    local_interfaces);
        std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                  << " dnoe update impl  \n";

        auto ret = std::make_shared<RepartDistMatrix>(
            exec, comm, repartitioner.get_repart_dim(), dist_A->get_size(),
            std::move(dist_A), local_sparsity_, non_local_sparsity_,
            src_comm_pattern, local_interfaces);

        return ret;
    };

    void update(const ExecutorHandler &exec_handler, word matrix_format,
                const Repartitioner &repartitioner,
                std::shared_ptr<const HostMatrixWrapper> host_A)
    {
        update_impl(exec_handler, matrix_format, repartitioner, host_A,
                    dist_mtx_, local_sparsity_, non_local_sparsity_,
                    src_comm_pattern_, local_interfaces_);
    };

    // TODO matrix_format can be data member
    static void update_impl(
        const ExecutorHandler &exec_handler, word matrix_format,
        const Repartitioner &repartitioner,
        std::shared_ptr<const HostMatrixWrapper> host_A,
        std::shared_ptr<
            gko::experimental::distributed::Matrix<scalar, label, label>>
            dist_A,
        std::shared_ptr<const SparsityPattern> local_sparsity,
        std::shared_ptr<const SparsityPattern> non_local_sparsity,
        std::shared_ptr<const CommunicationPattern> src_comm_pattern,
        std::vector<std::pair<bool, label>> local_interfaces)
    {
        auto exec = exec_handler.get_ref_exec();
        auto device_exec = exec_handler.get_device_exec();
        auto ranks_per_gpu = repartitioner.get_ranks_per_gpu();
        bool requires_host_buffer = exec_handler.get_gko_force_host_buffer();

        label rank{repartitioner.get_rank(exec_handler)};
        label owner_rank = repartitioner.get_owner_rank(exec_handler);
        bool owner = repartitioner.is_owner(exec_handler);
        label nrows = host_A->get_local_nrows();
        label local_matrix_nnz = host_A->get_local_matrix_nnz();
        // size + padding has to be local_matrix_nnz
        auto diag_comm_pattern = compute_send_recv_counts(
            exec_handler, ranks_per_gpu, nrows, local_matrix_nnz,
            local_matrix_nnz - nrows, 0);
        std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                  << " diag comm pattern  \n";


        label upper_nnz = host_A->get_upper_nnz();
        auto upper_comm_pattern = compute_send_recv_counts(
            exec_handler, ranks_per_gpu, upper_nnz, local_matrix_nnz, 0,
            local_matrix_nnz - upper_nnz);
        std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                  << " upper comm pattern  \n";
        auto lower_comm_pattern =
            compute_send_recv_counts(exec_handler, ranks_per_gpu, upper_nnz,
                                     local_matrix_nnz, upper_nnz, nrows);

        scalar *local_ptr;
        scalar *local_ptr_2;
        label nnz = 0;

        // update main values
        std::vector<scalar> loc_buffer;
        if (owner) {
            using Coo = gko::matrix::Coo<scalar, label>;
            auto local_mtx = dist_A->get_local_matrix();


            std::shared_ptr<const Coo> local =
                gko::as<Coo>(gko::as<CombinationMatrix<scalar, label, Coo>>(
                                 dist_A->get_local_matrix())
                                 ->get_combination()
                                 ->get_operators()[0]);
            if (requires_host_buffer) {
                loc_buffer.resize(local->get_num_stored_elements());
                local_ptr = loc_buffer.data();
                local_ptr_2 = const_cast<scalar *>(local->get_const_values());
            } else {
                local_ptr = const_cast<scalar *>(local->get_const_values());
            }
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
        std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                  << " done comm local mtx \n";

        if (requires_host_buffer) {
            auto host_buffer_view =
                gko::array<scalar>::view(exec, nnz, local_ptr);
            auto target_buffer_view =
                gko::array<scalar>::view(device_exec, nnz, local_ptr_2);
            target_buffer_view = host_buffer_view;
        }
        std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                  << " done copy to device  \n";

        // copy interface values
        auto comm = *exec_handler.get_communicator().get();
        if (owner) {
            using Coo = gko::matrix::Coo<scalar, label>;
            std::shared_ptr<const Coo> mtx;
            label loc_ctr{1};
            label nloc_ctr{0};
            label host_interface_ctr{0};
            label tag = 0;
            label comm_rank, comm_size;
            scalar *recv_buffer_ptr;
            std::vector<scalar> host_recv_buffer;
            label remain_host_interfaces = host_A->get_interface_size();
            for (auto [is_local, comm_rank] : local_interfaces) {
                label &ctr = (is_local) ? loc_ctr : nloc_ctr;
                if (is_local) {
                    mtx = gko::as<Coo>(
                        gko::as<CombinationMatrix<scalar, label, Coo>>(
                            dist_A->get_local_matrix())
                            ->get_combination()
                            ->get_operators()[ctr]);
                    comm_size = local_sparsity->interface_spans[ctr].length();
                } else {
                    mtx = gko::as<Coo>(
                        gko::as<CombinationMatrix<scalar, label, Coo>>(
                            dist_A->get_non_local_matrix())
                            ->get_combination()
                            ->get_operators()[ctr]);
                    comm_size =
                        non_local_sparsity->interface_spans[ctr].length();
                }

                if (requires_host_buffer) {
                    host_recv_buffer.resize(comm_size);
                    recv_buffer_ptr = host_recv_buffer.data();
                } else {
                    recv_buffer_ptr =
                        const_cast<scalar *>(mtx->get_const_values());
                }

                if (comm_rank != rank) {
                    std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                              << " comm_rank " << comm_rank << " rank " << rank
                              << "  \n";

                    comm.recv(exec, recv_buffer_ptr, comm_size, comm_rank, tag);
                } else {
                    // if data is already on this rank
                    auto data_view = gko::array<scalar>::const_view(
                        exec, comm_size,
                        host_A->get_interface_data(host_interface_ctr));

                    // TODO FIXME this needs target executor
                    auto target_view = gko::array<scalar>::view(
                        exec, comm_size, recv_buffer_ptr);

                    target_view = data_view;

                    host_interface_ctr++;
                    remain_host_interfaces--;
                }

                ctr++;
                // interface values need to be multiplied by -1
                using vec = gko::matrix::Dense<ValueType>;
                auto neg_one = gko::initialize<vec>({-1.0}, exec);
                auto interface_dense = vec::create(
                    exec, gko::dim<2>{comm_size, 1},
                    gko::array<scalar>::view(exec, comm_size, recv_buffer_ptr),
                    1);

                interface_dense->scale(neg_one);
            }
        } else {
            // the non owner has send all its interfaces to owner
            // thus all values need to be communicated to the owner as well
            label num_interfaces = src_comm_pattern->target_ids.get_size();
            label tag = 0;
            for (int i = 0; i < num_interfaces; i++) {
                label comm_size =
                    src_comm_pattern->target_sizes.get_const_data()[i];
                const scalar *send_buffer_ptr = host_A->get_interface_data(i);
                comm.send(exec, send_buffer_ptr, comm_size, owner_rank, tag);
            }
        }

        std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                  << " reorder \n";
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
                device_exec,
                gko::dim<2>{static_cast<dim_type>(local_elements), 1},
                gko::array<scalar>::view(device_exec, local_elements,
                                         local_ptr),
                1));
            std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                      << " local_elements " << local_elements << " reorder \n";

            auto mapping_view = gko::array<label>::view(
                exec, local_elements, local_sparsity->ldu_mapping.get_data());
            std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                      << " reorder \n";


            // TODO this needs to copy ldu_mapping to the device
            auto dense_vec = row_collection->clone();
            // auto dense_vec =
            // gko::share(gko::matrix::Dense<scalar>::create(exec,
            // row_collection->get_size()));

            std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                      << " reorder \n";
            dense_vec->row_gather(&mapping_view, row_collection.get());
            std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                      << " reorder \n";
        }
        std::cout << __FILE__ << ":" << __LINE__ << " rank " << rank
                  << "done reorder \n";
    };

    RepartDistMatrix(
        std::shared_ptr<const gko::Executor> exec, communicator comm,
        gko::dim<2> local_size, gko::dim<2> global_size,
        std::shared_ptr<dist_mtx> dist_mtx,
        std::shared_ptr<const SparsityPattern> local_sparsity,
        std::shared_ptr<const SparsityPattern> non_local_sparsity,
        std::shared_ptr<const CommunicationPattern> src_comm_pattern,
        std::vector<std::pair<bool, label>> &local_interfaces)
        : gko::experimental::EnableDistributedLinOp<RepartDistMatrix>(exec),
          gko::experimental::distributed::DistributedBase(comm),
          dist_mtx_(std::move(dist_mtx)),
          local_sparsity_(local_sparsity),
          non_local_sparsity_(non_local_sparsity),
          src_comm_pattern_(src_comm_pattern),
          local_interfaces_(local_interfaces)
    {
        this->set_size(global_size);
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

    // Needed for distributed/polymorphic_object.hpp
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
    std::shared_ptr<dist_mtx> dist_mtx_;

    std::shared_ptr<const SparsityPattern> local_sparsity_;

    std::shared_ptr<const SparsityPattern> non_local_sparsity_;

    std::shared_ptr<const CommunicationPattern> src_comm_pattern_;

    std::vector<std::pair<bool, label>> local_interfaces_;
};
