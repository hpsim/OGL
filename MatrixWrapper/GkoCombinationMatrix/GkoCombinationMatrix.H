#include <ginkgo/ginkgo.hpp>


// The CombinationMatrix class is a wrapper around Ginkgos combination linear
// operator. It provides an interface for creating linear combinations of a
// giveen inner matrix type and a read implementation
template <typename ValueType, typename IndexType, typename InnerMatrixType>
class CombinationMatrix
    : public gko::EnableLinOp<CombinationMatrix<ValueType>>,
      public gko::EnableCreateMethod<CombinationMatrix<ValueType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
public:
    // This constructor will be called by the create method.
    // Here we initialize, the interface_ranges to create the
    // the linear combination during the read call
    CombinationMatrix(std::shared_ptr<const gko::Executor> exec,
                      std::vector<gko::span> interface_ranges)
        : gko::EnableLinOp<CombinationMatrix>(exec, gko::dim<2>{size}),
          interface_ranges_(interface_ranges),
          comb_(gko::share(gko::Combination<ValueType>::create(exec, size)))
    {}

protected:
    using vec = gko::matrix::Dense<ValueType>;
    using coef_type = gko::array<ValueType>;

    void read(const device_matrix_data &data) override
    {
        auto exec = this->get_executor();
        for (auto &interface : interface_ranges_) {
            auto op gko::share(InnerMatrixType::create(this->exec))

                auto sub_array_view = [interface, exec](auto array_data) {
                    make_const_array_view(exec, interface.length(),
                                          array_data + interface.begin())
                };

            auto row_idxs = sub_array_view(data.get_const_row_idxs());
            auto col_idxs = sub_array_view(data.get_const_col_idxs());
            auto vals = sub_array_view(data.get_const_values());

            device_matrix_data in(exec, this->get_size(), row_idxs, col_idxs,
                                  vals);

            op->read(std::move(interface_view))

                combination->add_operators(
                    gko::initialize<gko::matrix::Dense<ValueType>>({1}, exec),
                    in);
        }
    }

    // Here we implement the application of the linear operator, x = A * b.
    // apply_impl will be called by the apply method, after the arguments have
    // been moved to the correct executor and the operators checked for
    // conforming sizes.
    //
    // For simplicity, we assume that there is always only one right hand side
    // and the stride of consecutive elements in the vectors is 1 (both of these
    // are always true in this example).
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override {
        this->comb_->apply_impl(b,x);
    }

    // There is also a version of the apply function which does the operation
    // x = alpha * A * b + beta * x. This function is commonly used and can
    // often be better optimized than implementing it using x = A * b. However,
    // for simplicity, we will implement it exactly like that in this example.
    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {
        this->comb_->apply_impl(alpha,b, beta, x);
    }

private:
    std::vector<gko::span> interface_ranges_;
    std::shared_ptr<gko::Combination> comb_;
};
