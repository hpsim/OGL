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
template <typename LocalMatrixType>
class RepartDistMatrix
    : public gko::experimental::EnableDistributedLinOp<
          RepartDistMatrix<LocalMatrixType>>,
      public gko::EnableCreateMethod<RepartDistMatrix<LocalMatrixType>>,
      public gko::experimental::distributed::DistributedBase {
    friend class gko::EnableCreateMethod<RepartDistMatrix<LocalMatrixType>>;
    friend class gko::EnablePolymorphicObject<RepartDistMatrix<LocalMatrixType>,
                                              gko::LinOp>;

    using dist_mtx =
        gko::experimental::distributed::Matrix<scalar, label, label>;
    using part_type = gko::experimental::distributed::Partition<label, label>;
    using vec = gko::matrix::Dense<scalar>;
    using device_matrix_data = gko::device_matrix_data<scalar, label>;
    using communicator = gko::experimental::mpi::communicator;

public:
    using gko::experimental::EnableDistributedLinOp<
        RepartDistMatrix<LocalMatrixType>>::convert_to;
    using gko::experimental::EnableDistributedLinOp<
        RepartDistMatrix<LocalMatrixType>>::move_to;

    std::shared_ptr<const gko::LinOp> get_local_matrix()
    {
        return this->dist_mtx_->get_local_matrix();
    }

    std::shared_ptr<const gko::LinOp> get_non_local_matrix()
    {
        return this->dist_mtx_->get_non_local_matrix();
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

    static std::shared_ptr<gko::LinOp> create(
        const ExecutorHandler &exec_handler, const Repartitioner &repartitioner,
        std::shared_ptr<const HostMatrixWrapper> host_A);

    void update(const ExecutorHandler &exec_handler,
                const Repartitioner &repartitioner,
                std::shared_ptr<const HostMatrixWrapper> host_A);

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

    void write(const word field_name_, const objectRegistry &db_) const;

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


std::shared_ptr<gko::LinOp> create_distributed(
    const ExecutorHandler &exec_handler, const Repartitioner &repartitioner,
    std::shared_ptr<const HostMatrixWrapper> hostMatrix, word matrix_format);
