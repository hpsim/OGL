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

#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/ginkgo.hpp>


#include "MatrixWrapper/CommunicationPattern/CommunicationPattern.H"
#include "MatrixWrapper/LDUMatrix/HostMatrix.H"


template <typename DistMtxType>
std::unique_ptr<DistMtxType> generate_dist_mtx_with_inner_type(
    word matrix_format, std::shared_ptr<gko::Executor> exec,
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


    std::shared_ptr<part_type> get_partition() { return this->partition_; }

    static std::shared_ptr<RepartDistMatrix> create(
        word matrix_format, std::shared_ptr<const HostMatrixWrapper> host_A)
    {
        auto exec_handler = host_A.get()->get_exec_handler();
        auto exec = exec_handler.get_device_exec();
        auto comm = *exec_handler.get_communicator().get();
        // TODO FIXME read from dictionary
        label ranks_per_gpu = 2;

        // repartition things here first by creating device_matrix_data on
        // device with correct size on each rank
        //
        // NOTE Start by copying first and repartition later
        auto local_sparsity_ = host_A->compute_local_sparsity(exec);
        auto non_local_sparsity_ = host_A->compute_non_local_sparsity(exec);

        auto orig_partition = gko::share(
            gko::experimental::distributed::build_partition_from_local_size<
                label, label>(exec, comm, local_sparsity_->dim[0]));

        // create original communicator pattern
        auto src_comm_pattern =
            host_A.get()->create_communication_pattern(comm);
        // create partiton here and pass to constructor
        auto dst_comm_pattern = repartition_comm_pattern(
            ranks_per_gpu, src_comm_pattern, orig_partition);

        label rank{src_comm_pattern.comm.rank()};

        auto repartition_sparsity =
            [ranks_per_gpu, src_comm_pattern, exec, orig_partition](
                std::shared_ptr<SparsityPattern> src_local_pattern,
                std::shared_ptr<SparsityPattern> src_non_local_pattern) {
                // 1. obtain send recv sizes vector
                // here we can reuse code from repartition_comm_pattern
                //
                // 2. initialize send and recv buffer
                // if we keep interfaces separated row and col indices can be
                //
                // 3. ldu mapping needs to be update?
                // 4. dim needs to be updated
                // this is not implemented yet, but we can't fail here for
                // debugging reasons
                // FatalErrorInFunction << "Not implemented" <<
                // abort(FatalError);
                auto target_rank = [ranks_per_gpu](label rank) {
                    return rank - (rank % ranks_per_gpu);
                };


                auto comm = src_comm_pattern.comm;

                // find ranks which are now local
                // the target_ids, target_sizes etc need to be communicated
                // with new owner. thus we use maps to store them where the
                // index is the new owner.
                label rank{src_comm_pattern.comm.rank()};
                label owner_rank = target_rank(rank);
                bool owner = rank == owner_rank;

                auto local_comm_pattern = compute_send_recv_counts(
                    ranks_per_gpu, owner_rank, src_local_pattern->size_, comm,
                    exec);

                auto gather_to_owner = [&](label offset, label *data,
                                           label size, auto &comm_pattern) {
                    if (offset > 0) {
                        std::transform(data, data + size, data,
                                       [&](label idx) { return idx + offset; });
                    }

                    auto [send_counts, recv_counts, send_offsets, recv_offsets,
                          recv_buffer] = comm_pattern;

                    comm.all_to_all_v(exec, data, send_counts.data(),
                                      send_offsets.data(), recv_buffer.data(),
                                      recv_counts.data(), recv_offsets.data());

                    return std::vector<label>{recv_buffer};
                };


                label offset = orig_partition->get_range_bounds()[rank] -
                               orig_partition->get_range_bounds()[owner_rank];

                std::vector<label> merged_local_rows = gather_to_owner(
                    offset, src_local_pattern->row_idxs.get_data(),
                    src_local_pattern->row_idxs.get_size(), local_comm_pattern);

                std::vector<label> merged_local_cols = gather_to_owner(
                    offset, src_local_pattern->col_idxs.get_data(),
                    src_local_pattern->row_idxs.get_size(), local_comm_pattern);

                std::vector<label> merged_ldu_mapping = gather_to_owner(
                    0, src_local_pattern->ldu_mapping.get_data(),
                    src_local_pattern->row_idxs.get_size(), local_comm_pattern);


                label rows = (owner) ? merged_local_rows.back() + 1 : 0;
                gko::dim<2> merged_local_dim{rows, rows};

                std::vector<label> merged_begin{0};
                std::vector<label> merged_end{ merged_local_rows.size()};
                std::vector<label> merged_ranks{rank};

                // TODO remove
                // std::cout << __FILE__ << " on rank " << rank
                //           << " merged_local_rows " << merged_local_rows
                //            << " merged_local_cols " << merged_local_cols
                //         << "\n";

                auto non_local_comm_pattern = compute_send_recv_counts(
                    ranks_per_gpu, owner_rank, src_non_local_pattern->size_,
                    comm, exec);

                std::vector<label> merged_non_local_rows = gather_to_owner(
                    offset, src_non_local_pattern->row_idxs.get_data(),
                    src_non_local_pattern->row_idxs.get_size(),
                    non_local_comm_pattern);

                std::vector<label> merged_non_local_cols = gather_to_owner(
                    offset, src_non_local_pattern->col_idxs.get_data(),
                    src_non_local_pattern->row_idxs.get_size(),
                    non_local_comm_pattern);

                std::vector<label> merged_non_local_ldu_mapping =
                    gather_to_owner(
                        0, src_non_local_pattern->ldu_mapping.get_data(),
                        src_non_local_pattern->row_idxs.get_size(),
                        non_local_comm_pattern);

                // TODO remove
                // std::cout << __FILE__ << " on rank " << rank
                //           << " merged_non_local_rows" <<
                //           merged_non_local_rows
                //           << " merged_non_local_cols " <<
                //           merged_non_local_cols
                //           << "\n";

                auto [span_send_counts, span_recv_counts, span_send_offsets,
                      span_recv_offsets, span_recv_buffer] =
                    compute_send_recv_counts(
                        ranks_per_gpu, owner_rank,
                        src_non_local_pattern->interface_spans.size(), comm,
                        exec);

                std::vector<label> spans_begin;
                std::vector<label> spans_end;

                for (auto elem : src_non_local_pattern->interface_spans) {
                    spans_begin.push_back(elem.begin);
                    spans_end.push_back(elem.end);
                }

                comm.all_to_all_v(
                    exec, spans_begin.data(), span_send_counts.data(),
                    span_send_offsets.data(), span_recv_buffer.data(),
                    span_recv_counts.data(), span_recv_offsets.data());

                std::vector<label> merged_non_local_begin{span_recv_buffer};

                comm.all_to_all_v(
                    exec, spans_end.data(), span_send_counts.data(),
                    span_send_offsets.data(), span_recv_buffer.data(),
                    span_recv_counts.data(), span_recv_offsets.data());
                std::vector<label> merged_non_local_end{span_recv_buffer};

                comm.all_to_all_v(
                    exec, src_non_local_pattern->rank.data(),
                    span_send_counts.data(), span_send_offsets.data(),
                    span_recv_buffer.data(), span_recv_counts.data(),
                    span_recv_offsets.data());
                std::vector<label> merged_non_local_ranks{span_recv_buffer};

                std::cout << __FILE__ << " on rank " << rank << " merged_begin"
                          << merged_begin << " merged_end " << merged_end
                          << " merged_ranks " << merged_non_local_ranks << "\n";

                // TODO shorten this
                std::vector<label> gathered_non_local_rows;
                std::vector<label> gathered_non_local_cols;
                std::vector<label> gathered_non_local_mapping;
                std::vector<label> gathered_ranks;
                std::vector<label> gathered_begin;
                std::vector<label> gathered_end;
                label merged_ranks_size = merged_non_local_ranks.size();
                for (label i=0;i<merged_ranks_size;i++) {
                    auto merged_target_rank = merged_non_local_ranks[i];
                    label to_rank = target_rank(merged_target_rank);
                    bool local = rank == to_rank;
                    std::cout << __FILE__ << " on rank " << rank
                              << " merged_target_rank " << merged_target_rank
                              << " is local " << local << " target_rank "
                              << to_rank << "\n";
                    auto begin = merged_begin[i];
                    auto end = merged_end[i];
                    if (local) {
                        label size_before = merged_local_rows.size();
                        merged_local_rows.insert(
                            merged_local_rows.end(),
                            merged_non_local_rows.begin() + begin,
                            merged_non_local_rows.begin() + end);
                        merged_local_cols.insert(
                            merged_local_cols.end(),
                            merged_non_local_cols.begin() + begin,
                            merged_non_local_cols.begin() + end);
                        merged_ldu_mapping.insert(
                            merged_ldu_mapping.end(),
                            merged_non_local_ldu_mapping.data() + begin,
                            merged_non_local_ldu_mapping.data() + end);

                        label size_after = merged_local_rows.size();
                        merged_begin.push_back(size_before);
                        merged_end.push_back(size_after);
                        merged_ranks.push_back(rank);
                    } else {
                        label size_before = gathered_non_local_rows.size();
                        gathered_non_local_rows.insert(
                            gathered_non_local_rows.end(),
                            merged_non_local_rows.begin() + begin,
                            merged_non_local_rows.begin() + end);
                        gathered_non_local_cols.insert(
                            gathered_non_local_cols.end(),
                            merged_non_local_cols.begin() + begin,
                            merged_non_local_cols.begin() + end);
                        gathered_non_local_mapping.insert(
                            gathered_non_local_mapping.end(),
                            merged_non_local_cols.begin() + begin,
                            merged_non_local_cols.begin() + end);
                        label size_after = gathered_non_local_rows.size();
                        std::cout << __FILE__ << " on rank " << rank
                                << " non_local size_before " << size_before
                                << " non_local size_after " << size_after
                                << "\n";
                        gathered_begin.push_back(size_before);
                        gathered_end.push_back(size_after);
                        gathered_ranks.push_back(to_rank);
                    }
                }

                // add interfaces
                // to add interfaces we go through the non_local_sparsity
                // pattern and check if interface is still non_local
                //
                // iterate interfaces and send to owner.
                // on owner decide to move to local or non_local
                // for sending to owner we can also use the all_to_all_v
                // approach we than have
                //
                // row [ 4 , 8,  12 | 16 , 17, 18 ] <- local row
                // col [ 1 , 2, 3 | 1, 2, 3 ] <- just the interface ctr
                // from repartition_comm_pattern we could get
                //
                // or we split local / non-local first
                // for this we could store ranges when computing
                // repartition_comm_pattern
                //


                if (owner) {
                    auto new_local_spars_pattern =
                        std::make_shared<SparsityPattern>(
                            exec, merged_local_rows.size(), merged_local_dim,
                            merged_local_rows, merged_local_cols,
                            merged_ldu_mapping, merged_begin, merged_end,
                            merged_ranks);

                    std::cout << __FILE__ << " on rank " << rank
                              << "local dim ["
                              << new_local_spars_pattern->dim[0] << " x "
                              << new_local_spars_pattern->dim[1] << "] \n";

                    auto new_non_local_spars_pattern =
                        std::make_shared<SparsityPattern>(
                            exec, gathered_non_local_rows.size(),
                            gko::dim<2>{merged_local_dim[0],
                                        gathered_non_local_rows.size()},
                            gathered_non_local_rows, gathered_non_local_cols,
                            gathered_non_local_mapping, gathered_begin,
                            gathered_end, gathered_ranks);

                    std::cout << __FILE__ << " on rank " << rank
                              << " done_sparsity \n";
                    return std::make_pair<std::shared_ptr<SparsityPattern>,
                                          std::shared_ptr<SparsityPattern>>(
                        std::move(new_local_spars_pattern),
                        std::move(new_non_local_spars_pattern));
                } else {
                    auto new_local_spars_pattern =
                        std::make_shared<SparsityPattern>(exec, 0);

                    auto new_non_local_spars_pattern =
                        std::make_shared<SparsityPattern>(exec, 0);

                    new_local_spars_pattern->dim = merged_local_dim;
                    new_non_local_spars_pattern->dim = merged_local_dim;

                    return std::make_pair<std::shared_ptr<SparsityPattern>,
                                          std::shared_ptr<SparsityPattern>>(
                        std::move(new_local_spars_pattern),
                        std::move(new_non_local_spars_pattern));
                }
            };

        auto [repart_loc_sparsity, repart_non_loc_sparsity] =
            repartition_sparsity(local_sparsity_, non_local_sparsity_);

        local_sparsity_ = repart_loc_sparsity;
        non_local_sparsity_ = repart_non_loc_sparsity;

        // const label local_nnz = local_sparsity_->size_;
        // const label non_local_nnz = non_local_sparsity_->size_;

        auto localized_partition = local_part_type::build_from_blocked_recv(
            exec, local_sparsity_->dim[0], dst_comm_pattern.send_idxs,
            dst_comm_pattern.target_ids, dst_comm_pattern.target_sizes);


        std::cout << __FILE__ << " rank " << rank << " local sparsity dim"
                  << local_sparsity_->dim[0] << "non_local sparsity dim"
                  << non_local_sparsity_->dim[0] << "x"
                  << non_local_sparsity_->dim[1] << " target_ids "
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

        // host_A.get()->compute_local_coeffs(local_sparsity_, local_coeffs);
        // host_A.get()->compute_non_local_coeffs(non_local_sparsity_,
        //                                        non_local_coeffs);

        std::cout << __FILE__ << " on rank " << rank
                  << " done compute local coeffs \n";
        dist_A->read_distributed(
            device_matrix_data(exec, local_sparsity_->dim,
                               local_sparsity_->row_idxs,
                               local_sparsity_->col_idxs, local_coeffs),
            device_matrix_data(exec, non_local_sparsity_->dim,
                               non_local_sparsity_->row_idxs,
                               non_local_sparsity_->col_idxs, non_local_coeffs),
            sparse_comm);
        std::cout << __FILE__ << " on rank " << rank
                  << " done read_distributed  dist_A size" << dist_A->get_size()
                  << " \n";

        return std::make_shared<RepartDistMatrix>(
            exec, comm, host_A->get_size(), dist_A->get_size(),
            std::move(dist_A));
    };

    // TODO matrix_format can be data member
    void update(word matrix_format,
                std::shared_ptr<const HostMatrixWrapper> host_A)
    {
        auto exec_handler = host_A.get()->get_exec_handler();
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
        partition_ = gko::share(
            gko::experimental::distributed::build_partition_from_local_size<
                label, label>(exec, comm, local_size[0]));
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

    RepartDistMatrix(std::shared_ptr<const gko::Executor> exec,
                     communicator comm)
        : gko::experimental::EnableDistributedLinOp<RepartDistMatrix>(exec),
          gko::experimental::distributed::DistributedBase{comm}
    {}

protected:
    // Here we implement the application of the linear operator, x = A * b.
    // apply_impl will be called by the apply method, after the arguments have
    // been moved to the correct executor and the operators checked for
    // conforming sizes.
    //
    // For simplicity, we assume that there is always only one right hand side
    // and the stride of consecutive elements in the vectors is 1 (both of these
    // are always true in this example).
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        this->dist_mtx_->apply(b, x);
    }


    // There is also a version of the apply function which does the operation
    // x = alpha * A * b + beta * x. This function is commonly used and can
    // often be better optimized than implementing it using x = A * b. However,
    // for simplicity, we will implement it exactly like that in this example.
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
