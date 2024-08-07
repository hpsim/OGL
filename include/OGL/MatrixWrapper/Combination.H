// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"

/* @brief The CombinationMatrix class is a wrapper around Ginkgos combination
 * linear operator. It provides an interface for creating linear combinations of
 * a given inner matrix type and a read implementation
 * */
template <typename InnerMatrixType>
class CombinationMatrix
    : public gko::EnableLinOp<CombinationMatrix<InnerMatrixType>>,
      public gko::EnableCreateMethod<CombinationMatrix<InnerMatrixType>>,
      public gko::ReadableFromMatrixData<scalar, label> {
      public gko::ConvertibleTo<gko::matrix::Coo<scalar, label>>,
      public gko::ConvertibleTo<gko::matrix::Csr<scalar, label>>,
    friend class gko::EnableCreateMethod<CombinationMatrix<InnerMatrixType>>;
    friend class gko::EnablePolymorphicObject<
        CombinationMatrix<InnerMatrixType>, gko::LinOp>;

public:
    using gko::EnableLinOp<CombinationMatrix<InnerMatrixType>>::convert_to;
    using gko::EnableLinOp<CombinationMatrix<InnerMatrixType>>::move_to;

    using vec = gko::matrix::Dense<scalar>;
    using coef_type = gko::array<scalar>;
    using device_matrix_data = gko::device_matrix_data<scalar, label>;
    using matrix_data = gko::matrix_data<scalar, label>;

    void read(const matrix_data &data) override
    {
        FatalErrorInFunction << "Not implemented" << abort(FatalError);
    }

    /* @brief Implements a reading from device_matrix_data
     *
     * This function implements read from matrix data. Here
     * matrix data is consecutive data and interface_ranges
     * is used to separated the individual submatrices
     * */
    void read(const device_matrix_data &data) override
    {
        auto exec = this->get_executor();
        auto data_exec = data.get_executor();

        for (auto &interface : ranges_) {
            auto length = interface.length();
            auto offset = interface.begin;
            auto op = gko::share(InnerMatrixType::create(exec));

            // create views into corresponding sections
            // of the data
            auto row_view = gko::array<label>::const_view(
                data_exec, length, data.get_const_row_idxs() + offset);

            auto col_view = gko::array<label>::const_view(
                data_exec, length, data.get_const_col_idxs() + offset);

            auto val_view = gko::array<scalar>::const_view(
                data_exec, length, data.get_const_values() + offset);

            // create a new device_matrix data  object from the views
            // this will create copies ow
            device_matrix_data in(
                data_exec, comb_->get_size(), row_view.copy_to_array(),
                col_view.copy_to_array(), val_view.copy_to_array());

            // this will move data to exec if exec != data_exec
            op->read(in);
            this->comb_->add_operators(
                gko::initialize<gko::matrix::Dense<scalar>>({1}, exec), op);
        }

        // NOTE find a better place
        this->set_size(comb_->get_size());

        // TODO validate the size, needs assertions
        // OGL_ASSERT(this->get_size()[0] == this.interface_ranges_.back().end);
    }

    std::shared_ptr<gko::Combination<scalar>> get_combination() const
    {
        return this->comb_;
    }

    /**
     * Copy-assigns a CombinationMatrix matrix. Preserves executor, copies
     * everything else.
     */
    CombinationMatrix &operator=(const CombinationMatrix &other)
    {
        if (&other != this) {
            gko::EnableLinOp<CombinationMatrix>::operator=(other);
            for (auto range : other.ranges_) {
                ranges_.emplace_back(range.begin, range.end);
            }
            comb_ = other.comb_;
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
            ranges_ = std::move(other.ranges_);
            comb_ = std::move(other.comb_);
        }
        return *this;
    }

    const std::vector<std::shared_ptr<const gko::LinOp>> &get_coefficients()
        const noexcept
    {
        return comb_->get_coefficients();
    }

    /**
     * Returns a list of operators of the combination.
     *
     * @return a list of operators
     */
    const std::vector<std::shared_ptr<const gko::LinOp>> &get_operators()
        const noexcept
    {
        return comb_->get_operators();
    }

    void convert_to(
        gko::matrix::Coo<scalar>, label>* result) const override {};

    void convert_to(
        gko::matrix::Csr<scalar>, label>* result) const override {};

protected:
    CombinationMatrix(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<CombinationMatrix>(exec)
    {}

    /* This constructor will be called by the create method.
     * Here we initialize, the interface_ranges to create the
     * the linear combination during the read call
     * @param executor where combitination matrix data resides on
     * @param the global size of the matrices, eg. max (row, col idxs)
     * @param ranges required for read to separate interfaces
     *        from consecutive matrix_data input. The ranges contain first and
     * last index of the individual matrix
     */
    CombinationMatrix(std::shared_ptr<const gko::Executor> exec,
                      gko::dim<2> size, std::vector<gko::span> ranges)
        : gko::EnableLinOp<CombinationMatrix>(exec),
          ranges_(ranges),
          comb_(gko::share(gko::Combination<scalar>::create(exec, size)))
    {}

    /* Forwarding call of apply which checks if the matrix contains any
     * submatrices
     * */
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        if (ranges_.size() > 0) {
            this->comb_->apply(b, x);
        }
    }

    /* Forwarding call of apply which checks if the matrix contains any
     * submatrices
     * */
    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {
        if (ranges_.size() > 0) {
            this->comb_->apply(alpha, b, beta, x);
        }
    }


private:
    std::vector<gko::span>
        ranges_;  // vector containing start and end indices of the
                  // individual matrices in the consecutive data
    std::shared_ptr<gko::Combination<scalar>> comb_;
};
