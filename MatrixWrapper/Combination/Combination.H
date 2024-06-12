#pragma once

#include <ginkgo/ginkgo.hpp>

// The CombinationMatrix class is a wrapper around Ginkgos combination linear
// operator. It provides an interface for creating linear combinations of a
// giveen inner matrix type and a read implementation
// template <typename ValueType, typename IndexType, typename gko::matrix::Coo>
template <typename ValueType, typename IndexType, typename InnerMatrixType>
class CombinationMatrix
    : public gko::EnableLinOp<
          CombinationMatrix<ValueType, IndexType, InnerMatrixType>>,
      public gko::EnableCreateMethod<
          CombinationMatrix<ValueType, IndexType, InnerMatrixType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<
        CombinationMatrix<ValueType, IndexType, InnerMatrixType>>;
    friend class gko::EnablePolymorphicObject<
        CombinationMatrix<ValueType, IndexType, InnerMatrixType>, gko::LinOp>;

public:
    using gko::EnableLinOp<
        CombinationMatrix<ValueType, IndexType, InnerMatrixType>>::convert_to;
    using gko::EnableLinOp<
        CombinationMatrix<ValueType, IndexType, InnerMatrixType>>::move_to;

    using vec = gko::matrix::Dense<ValueType>;
    using coef_type = gko::array<ValueType>;
    using device_matrix_data = gko::device_matrix_data<scalar, label>;
    using matrix_data = gko::matrix_data<scalar, label>;

    void read(const matrix_data &data) override {}

    void read(const device_matrix_data &data) override
    {
        // TODO FIXME if no matrix is given because of repartitioning
        // which might remove non_local matrix
        // a zero sized dummy matrix needs to be created
        auto exec = this->get_executor();
        auto data_exec = data.get_executor();

        for (auto &interface : interface_ranges_) {
            auto length = interface.length();
            auto offset = interface.begin;
            auto op = gko::share(InnerMatrixType::create(exec));

            device_matrix_data in(
                exec, comb_->get_size(),
                gko::array<IndexType>(
                    exec, data.get_const_row_idxs() + offset,
                    data.get_const_row_idxs() + offset + length),
                gko::array<IndexType>(
                    exec, data.get_const_col_idxs() + offset,
                    data.get_const_col_idxs() + offset + length),
                gko::array<ValueType>(
                    exec, data.get_const_values() + offset,
                    data.get_const_values() + offset + length));

            op->read(in);
            this->comb_->add_operators(
                gko::initialize<gko::matrix::Dense<ValueType>>({1}, exec), op);
        }
        // NOTE find a better place
        this->set_size(comb_->get_size());
    }

    std::shared_ptr<gko::Combination<ValueType>> get_combination() const
    {
        return this->comb_;
    }


    /**
     * Copy-assigns a CombinationMatrix matrix. Preserves executor, copies
     * everything else.
     */
    CombinationMatrix &operator=(const CombinationMatrix &other)
    {
        // FatalErrorInFunction << "Copying CombinationMatrix is not allowed"
        //                      << abort(FatalError);
        std::cout << __FILE__ << ":" << __LINE__ << "FIXME why is copy neeeded?\n";
        if (&other != this) {
            gko::EnableLinOp<CombinationMatrix>::operator=(other);
            for (auto range: other.interface_ranges_){
                interface_ranges_.emplace_back(range.begin, range.end);
            }
            comb_ = other.comb_;
            // restore other invariant
            // other.row_ptrs_.resize_and_reset(1);
        }
        return *this;
    }

    /**
     * Move-assigns a CombinationMatrix matrix. Preserves executor, moves the
     * data and leaves the moved-from object in an empty state (0x0 LinOp with
     * unchanged executor and strategy, no nonzeros and valid row pointers).
     */
    CombinationMatrix &operator=(CombinationMatrix &&other)
    {
        if (&other != this) {
            gko::EnableLinOp<CombinationMatrix>::operator=(std::move(other));
            interface_ranges_ = std::move(other.interface_ranges_);
            comb_ = std::move(other.comb_);
            // restore other invariant
            // other.row_ptrs_.resize_and_reset(1);
        }
        return *this;
    }


protected:
    CombinationMatrix(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<CombinationMatrix>(exec)
    {}

    // This constructor will be called by the create method.
    // Here we initialize, the interface_ranges to create the
    // the linear combination during the read call
    // TODO dont allow copying
    CombinationMatrix(std::shared_ptr<const gko::Executor> exec,
                      gko::dim<2> size, std::vector<gko::span> interface_ranges)
        : gko::EnableLinOp<CombinationMatrix>(exec),
          interface_ranges_(interface_ranges),
          comb_(gko::share(gko::Combination<ValueType>::create(exec, size)))
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
        if (interface_ranges_.size() > 0) {
            this->comb_->apply(b, x);
        }
    }

    // There is also a version of the apply function which does the operation
    // x = alpha * A * b + beta * x. This function is commonly used and can
    // often be better optimized than implementing it using x = A * b. However,
    // for simplicity, we will implement it exactly like that in this example.
    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {
        if (interface_ranges_.size() > 0) {
            this->comb_->apply(alpha, b, beta, x);
        }
    }


private:
    std::vector<gko::span> interface_ranges_;
    std::shared_ptr<gko::Combination<ValueType>> comb_;
};
