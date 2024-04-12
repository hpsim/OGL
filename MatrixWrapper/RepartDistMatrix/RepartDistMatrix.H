#include <ginkgo/ginkgo.hpp>


// The RepartDistMatrix class is a wrapper around Ginkgos distributed Matrix class
// 
// It adds functionality for repeated read and repartitioning operatitions. As a constraint
// it is required that the inner matrix types of the distributed matrix are of RepartDistMatrix type.
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
class RepartDistMatrix
    : public gko::EnableLinOp<
          RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>,
      public gko::EnableCreateMethod<
          RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>> {
    friend class gko::EnableCreateMethod<
        RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>;
    friend class gko::EnablePolymorphicObject<
        RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>, gko::LinOp>;

    using dist_mtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType, GlobalIndexType>;

public:
    using gko::EnableLinOp<
        RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>::convert_to;
    using gko::EnableLinOp<
        RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType>>::move_to;

    using vec = gko::matrix::Dense<ValueType>;
    using coef_type = gko::array<ValueType>;
    using device_matrix_data = gko::device_matrix_data<scalar, label>;
    using matrix_data = gko::matrix_data<scalar, label>;
    using sparse_communicator = gko::experimental::distributed::sparse_communicator;


    void read_distributed(
        const device_matrix_data& local_data,
        const device_matrix_data& non_local_data,
        std::shared_ptr<const sparse_communicator> sparse_comm)
    {
        this->dist_mtx_->read_distributed(local_data, non_local_data, sparse_comm);
    }

    std::shared_ptr<const gko::LinOp> get_non_local_matrix() {
        this->dist_mtx_->get_local_matrix();
    }

    std::shared_ptr<const gko::LinOp> get_local_matrix() {
        this->dist_mtx_->get_non_local_matrix();
    }

    // Function with the exact same interface as create but reuses intermediate computations 
    static std::unique_ptr<RepartDistMatrix<ValueType, LocalIndexType, GlobalIndexType> > create(
        std::shared_ptr<gko::Executor> exec,
        std::shared_ptr<gko::experimental::mpi::communicator> comm,
        std::shared_ptr<const gko::LinOp> local_template,
        std::shared_ptr<const gko::LinOp> non_local_template
        ); 

    // Function with the exact same interface as create but reuses intermediate computations 
    void update(
        std::shared_ptr<gko::Executor> exec,
        std::shared_ptr<gko::experimental::mpi::communicator> comm,
        std::shared_ptr<gko::LinOp> local_template,
        std::shared_ptr<gko::LinOp> non_local_template
            ) {}

    /**
     * Copy-assigns a CombinationMatrix matrix. Preserves executor, copies
     * everything else.
     */
    RepartDistMatrix &operator=(const RepartDistMatrix &other)
    {
        if (&other != this) {
            *dist_mtx_.get() = *other.dist_mtx_.get();
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
        }
        return *this;
    }

protected:
    RepartDistMatrix(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<RepartDistMatrix>(exec)
    {}

    RepartDistMatrix(std::shared_ptr<const gko::Executor> exec,
        std::shared_ptr<const sparse_communicator> sparse_comm,
        std::unique_ptr<const gko::LinOp> local_template,
        std::unique_ptr<const gko::LinOp> non_local_template
                      )
        : gko::EnableLinOp<RepartDistMatrix>(exec),
          dist_mtx_(dist_mtx::create(exec, sparse_comm, local_template, non_local_template))
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
};
