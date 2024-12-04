// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <map>

#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/ginkgo.hpp>

#include "OGL/CommunicationPattern.H"
#include "OGL/MatrixWrapper/Combination.H"
#include "OGL/MatrixWrapper/HostMatrix.H"
#include "OGL/Repartitioner.H"


/* The RepartDistMatrix class is a wrapper around Ginkgos distributed Matrix
 * class
 *
 * It adds functionality for repeated read and repartitioning operatitions. As a
 * constraint it is required that the inner matrix types of the distributed
 * matrix are of RepartDistMatrix type.
 * */
class RepartDistMatrix
    : public gko::experimental::EnableDistributedLinOp<RepartDistMatrix>,
      public gko::EnableCreateMethod<RepartDistMatrix>,
      public gko::experimental::distributed::DistributedBase {
    friend class gko::EnableCreateMethod<RepartDistMatrix>;
    friend class gko::EnablePolymorphicObject<RepartDistMatrix, gko::LinOp>;

public:
    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;
    using part_type = gko::experimental::distributed::Partition<label, label>;
    using vec = gko::matrix::Dense<scalar>;
    using device_matrix_data = gko::device_matrix_data<scalar, label>;
    using communicator = gko::experimental::mpi::communicator;
    // id, comm_pattern, data_ptr
    using all_to_all_data = std::tuple<label, AllToAllPattern, scalar *>;
    struct pairwise_data {
        label id;         // original interface id on orig rank
        label send;       // 0 - send, 1, receive, 2 same_rank
        label comm_rank;  // other side of communication
        label length;
        label send_id;
        scalar *recv_ptr;
    };

    using gko::experimental::EnableDistributedLinOp<
        RepartDistMatrix>::convert_to;
    using gko::experimental::EnableDistributedLinOp<RepartDistMatrix>::move_to;

    std::shared_ptr<const gko::LinOp> get_local_matrix() const
    {
        return this->dist_mtx_->get_local_matrix();
    }

    std::shared_ptr<const gko::LinOp> get_non_local_matrix() const
    {
        return this->dist_mtx_->get_non_local_matrix();
    }

    std::shared_ptr<const gko::LinOp> get_local() const
    {
        if (fuse_) {
            return dist_mtx_->get_local_matrix();
        } else {
            return gko::as<CombinationMatrix<gko::LinOp>>(
                       dist_mtx_->get_local_matrix())
                ->get_operators()[0];
        }
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
                RepartDistMatrix>::operator=(other);
            this->dist_mtx_ = other.dist_mtx_;
            this->fuse_ = other.fuse_;
            this->matrix_format_ = other.matrix_format_;
            this->repartitioner_ = other.repartitioner_;
            this->all_to_all_update_data_ = other.all_to_all_update_data_;
            this->pairwise_update_data_ = other.pairwise_update_data_;
            this->reorder_maps_ = other.reorder_maps_;
            this->compress_to_global_ = other.compress_to_global_;
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
            this->fuse_ = other.fuse_;
            this->matrix_format_ = other.matrix_format_;
            this->dist_mtx_ = std::move(other.dist_mtx_);
            this->repartitioner_ = std::move(other.repartitioner_);
            this->all_to_all_update_data_ =
                std::move(other.all_to_all_update_data_);
            this->pairwise_update_data_ =
                std::move(other.pairwise_update_data_);
            this->reorder_maps_ = std::move(other.reorder_maps_);
            this->compress_to_global_ = std::move(other.compress_to_global_);
        }
        return *this;
    }

    template <typename InnerType>
    void update(const ExecutorHandler &exec_handler,
                std::shared_ptr<const HostMatrixWrapper> host_A, label verbose);

    word get_matrix_format() const { return matrix_format_; }

    std::shared_ptr<const gko::LinOp> get_dist_mtx() const { return dist_mtx_; }

    RepartDistMatrix(
        std::shared_ptr<const gko::Executor> exec, communicator comm,
        word matrix_format, std::shared_ptr<dist_mtx> dist_mtx,
        std::shared_ptr<const Repartitioner> repartitioner, bool fuse,
        std::vector<all_to_all_data> all_to_all_update_data,
        std::vector<pairwise_data> pairwise_update_data,
        std::vector<std::tuple<std::shared_ptr<gko::array<label>>, scalar *>>
            reorder_maps,
        std::vector<label> compress_to_global)
        : gko::experimental::EnableDistributedLinOp<RepartDistMatrix>(exec),
          gko::experimental::distributed::DistributedBase(comm),
          fuse_(fuse),
          matrix_format_(matrix_format),
          dist_mtx_(std::move(dist_mtx)),
          all_to_all_update_data_(all_to_all_update_data),
          pairwise_update_data_(pairwise_update_data),
          repartitioner_(repartitioner),
          reorder_maps_(reorder_maps),
          compress_to_global_(compress_to_global)
    {
        this->set_size(dist_mtx_->get_size());
    }

    template <typename LocalMatrixType>
    void write(const ExecutorHandler &exec_handler, const word field_name_,
               const objectRegistry &db_, bool write_global) const;

    // Needed for distributed/polymorphic_object.hpp
    RepartDistMatrix(std::shared_ptr<const gko::Executor> exec,
                     communicator comm)
        : gko::experimental::EnableDistributedLinOp<RepartDistMatrix>(exec),
          gko::experimental::distributed::DistributedBase{comm}
    {}

    std::shared_ptr<const Repartitioner> get_repartitioner() const
    {
        return repartitioner_;
    }

    const std::vector<label> &compress_to_global() const
    {
        return compress_to_global_;
    }

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
    bool fuse_;

    word matrix_format_;

    std::shared_ptr<dist_mtx> dist_mtx_;

    std::vector<all_to_all_data> all_to_all_update_data_;

    std::vector<pairwise_data> pairwise_update_data_;

    std::shared_ptr<const Repartitioner> repartitioner_;

    std::vector<std::tuple<std::shared_ptr<gko::array<label>>, scalar *>>
        reorder_maps_;

    std::vector<label> compress_to_global_;
};

std::shared_ptr<const gko::LinOp> get_local(
    std::shared_ptr<const gko::LinOp> dist_A);

void write_distributed(const ExecutorHandler &exec_handler, word field_name,
                       const objectRegistry &db,
                       std::shared_ptr<RepartDistMatrix> dist_A,
                       bool write_global);

void update_distributed(const ExecutorHandler &exec_handler,
                        std::shared_ptr<const HostMatrixWrapper> host_A,
                        std::shared_ptr<RepartDistMatrix> dist_A,
                        word matrix_format, label verbose);

std::shared_ptr<RepartDistMatrix> create_distributed(
    const ExecutorHandler &exec_handler,
    std::shared_ptr<const Repartitioner> repartitioner,
    std::shared_ptr<const HostMatrixWrapper> hostMatrix, word matrix_format,
    bool fuse, label verbose);
