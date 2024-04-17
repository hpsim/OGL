#include <ginkgo/ginkgo.hpp>

#include "MatrixWrapper/LDUMatrix/HostMatrix.H"

// The RepartDistMatrix class is a wrapper around Ginkgos distributed Matrix
// class
//
// It adds functionality for repeated read and repartitioning operatitions. As a
// constraint it is required that the inner matrix types of the distributed
// matrix are of RepartDistMatrix type.
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
class RepartDistMatrix
    : public gko::EnableLinOp<
          RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>,
      public gko::EnableCreateMethod<
          RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>> {
    friend class gko::EnableCreateMethod<
        RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>;
    friend class gko::EnablePolymorphicObject<
        RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>,
        gko::LinOp>;

    using dist_mtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                               GlobalIndexType>;

    using part_type =
        gko::experimental::distributed::Partition<label, label>;

    using localized_partition =
        gko::experimental::distributed::localized_partition<label>;

public:
    using gko::EnableLinOp<RepartDistMatrix<ValueType, LocalIndexType,
                                            GlobalIndexType>>::convert_to;
    using gko::EnableLinOp<
        RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>::move_to;

    using vec = gko::matrix::Dense<ValueType>;
    using coef_type = gko::array<ValueType>;
    using device_matrix_data = gko::device_matrix_data<scalar, label>;
    using matrix_data = gko::matrix_data<scalar, label>;
    using communicator = gko::experimental::mpi::communicator;
    using sparse_communicator =
        gko::experimental::distributed::sparse_communicator;


    void read_distributed(
        const device_matrix_data &local_data,
        const device_matrix_data &non_local_data,
        std::shared_ptr<const sparse_communicator> sparse_comm)
    {
        this->dist_mtx_->read_distributed(local_data, non_local_data,
                                          sparse_comm);
        this->set_size(dist_mtx_->get_size());
    }

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
            *dist_mtx_.get() = *other.dist_mtx_.get();
            this->set_size(dist_mtx_->get_size());
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
            gko::EnableLinOp<RepartDistMatrix>::operator=(std::move(other));
            dist_mtx_ = std::move(other.dist_mtx_);
            this->set_size(dist_mtx_->get_size());
        }
        return *this;
    }

    std::shared_ptr<dist_mtx> generate_dist_mtx_with_inner_type(
        word matrix_format, std::shared_ptr<gko::Executor> exec,
        std::shared_ptr<gko::experimental::mpi::communicator> comm) const
    {
        if (matrix_format == "Csr") {
            return dist_mtx::create(exec, *comm.get(),
                                    gko::with_matrix_type<gko::matrix::Csr>());
        }
        if (matrix_format == "Ell") {
            return dist_mtx::create(exec, *comm.get(),
                                    gko::with_matrix_type<gko::matrix::Ell>());
        }
        if (matrix_format == "Coo") {
            return dist_mtx::create(exec, *comm.get(),
                                    gko::with_matrix_type<gko::matrix::Coo>());
        }

        if (matrix_format == "Ldu") {
        }

        FatalErrorInFunction
            << "Matrix format " << matrix_format
            << " not supported! Supported formats: Csr, Ell, Coo"
            << abort(FatalError);
    }

    std::shared_ptr<part_type> get_partition() {return partition_;}

    static std::shared_ptr<RepartDistMatrix> create(
        word matrix_format, std::shared_ptr<const HostMatrixWrapper> host_A )
    {
        using part_type =
            gko::experimental::distributed::localized_partition<label>;

        auto exec_handler = host_A.get()->get_exec_handler();
        auto exec = exec_handler.get_device_exec();

        // repartition things here first by creating device_matrix_data on
        // device with correct size on each rank
        //
        // NOTE Start by copying first and repartition later
        auto local_sparsity_pattern = host_A->compute_local_sparsity(exec);
        auto non_local_sparsity_pattern = host_A->compute_non_local_sparsity(exec);

        // create original communicator pattern
        auto src_comm_pattern = host_A.get()->create_communication_pattern();
        // auto dst_comm_pattern = repartition_comm_pattern(src_comm_pattern);

        std::cout << "!!!!! END\n";

        // // if accumulate_ranks
        // // 1. move to shared buffer
        // // 2. switch interfaces to local_data
        // // 3. modify communication pattern
        // auto [local_sparsity_pattern_r, non_local_sparsity_pattern_r] =
        //     repartition_indices(ranks_per_gpu, source_partition,
        //                         local_sparsity_pattern,
        //                         non_local_sparsity_pattern);
        //
        // auto [local_coeffs, non_local_coeffs] = host_A->compute_coeffs();
        // auto [local_coeff_r, non_local_coeff_r] =
        //     repartition_coeffs(ranks_per_gpu, source_partition,
        //                        local_device_data, non_local_device_data);
        //
        //
        // partition = part_type::build_from_blocked_recv(
        //     exec, local_size_, comm_pattern.send_idxs,
        //     comm_pattern.target_ids, comm_pattern.target_sizes);
        //
        // auto sparse_comm =
        //     gko::experimental::distributed::sparse_communicator::create(
        //         *comm.get(), partition_.get_localized_partition());
        //
        // auto dist_A = generate_dist_mtx_with_inner_type(
        //     exec, host_A->get_communicator(), );
        //
        // // calls read of inner matrix type
        // dist_A->read_distributed(local_device_data, non_local_device_data,
        //                          sparse_comm);
        //
        // return dist_A;
    };

    void update(std::shared_ptr<const HostMatrixWrapper> host_A ){
        // On update dist_mtx already exists
        // and only local and non local coeffs need to be updated
        //
        // repartition_coeffs(local_device_data, non_local_device_data)
    };

    void write();

protected:
    RepartDistMatrix(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<RepartDistMatrix>(exec)
    {}

    RepartDistMatrix(std::shared_ptr<const gko::Executor> exec,
                     std::shared_ptr<const communicator> comm,
                     std::unique_ptr<const gko::LinOp> local_template,
                     std::unique_ptr<const gko::LinOp> non_local_template)
        : gko::EnableLinOp<RepartDistMatrix>(exec),
          dist_mtx_(dist_mtx::create(exec, *comm.get(), local_template,
                                     non_local_template))
    {}

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

};
