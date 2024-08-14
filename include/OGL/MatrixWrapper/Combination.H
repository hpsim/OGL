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
      public gko::ReadableFromMatrixData<scalar, label>,
      public gko::ConvertibleTo<gko::matrix::Coo<scalar, label>>,
      public gko::ConvertibleTo<gko::matrix::Csr<scalar, label>> {
    friend class gko::EnableCreateMethod<CombinationMatrix<InnerMatrixType>>;
    friend class gko::EnablePolymorphicObject<
        CombinationMatrix<InnerMatrixType>, gko::LinOp>;

public:
    using gko::EnableLinOp<CombinationMatrix<InnerMatrixType>>::convert_to;
    using gko::EnableLinOp<CombinationMatrix<InnerMatrixType>>::move_to;

    using matrix_data = gko::matrix_data<scalar, label>;

    void read(const matrix_data &data) override
    {
        FatalErrorInFunction << "Not implemented" << abort(FatalError);
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
            comb_ = std::move(other.comb_);
        }
        return *this;
    }

     /**
     * Returns a list of coefficients of the combination.
     *
     * @return a list of coefficients
     */
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

    /*
     * @warning assumes that all entries are unique
     */
    void convert_to(gko::matrix::Coo<scalar, label> *result) const override
    {
        auto exec = this->get_executor();
        auto operators = this->get_operators();

        std::vector<gko::matrix_data_entry<scalar, label>> data;
        for (auto op : operators) {
            auto tmp_mat_data = gko::matrix_data<scalar>();
            gko::as<InnerMatrixType>(op)->write(tmp_mat_data);
            data.insert(data.end(), tmp_mat_data.nonzeros.begin(),
                        tmp_mat_data.nonzeros.end());
        }
        auto mat_data = gko::matrix_data<scalar>();
        mat_data.nonzeros = data;
        mat_data.size = this->get_size();

        result->read(mat_data);
    }

    /*
     * @warning assumes that all entries are unique
     */
    void convert_to(gko::matrix::Csr<scalar, label> *result) const override 
    {
        auto exec = this->get_executor();
        auto operators = this->get_operators();

        std::vector<gko::matrix_data_entry<scalar, label>> data;
        for (auto op : operators) {
            auto tmp_mat_data = gko::matrix_data<scalar>();
            gko::as<InnerMatrixType>(op)->write(tmp_mat_data);
            data.insert(data.end(), tmp_mat_data.nonzeros.begin(),
                        tmp_mat_data.nonzeros.end());
        }
        auto mat_data = gko::matrix_data<scalar>();
        mat_data.nonzeros = data;
        mat_data.size = this->get_size();

        result->read(mat_data);
    }

    void move_to(gko::matrix::Coo<scalar, label> *result) override {}

    void move_to(gko::matrix::Csr<scalar, label> *result) override {}

protected:
    CombinationMatrix(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<CombinationMatrix>(exec),
          comb_(gko::share(gko::Combination<scalar>::create(exec)))
    {}

    CombinationMatrix(std::shared_ptr<const gko::Executor> exec,
                      gko::dim<2> size,
                      std::vector<std::shared_ptr<const gko::LinOp>> operators)
        : gko::EnableLinOp<CombinationMatrix>(exec),
          comb_(gko::share(gko::Combination<scalar>::create(exec, size)))
    {
        for (auto &op : operators) {
            this->comb_->add_operators(
                gko::initialize<gko::matrix::Dense<scalar>>({1}, exec), op);
        }
        this->set_size(size);
    }

    /* Forwarding call of apply which checks if the matrix contains any
     * submatrices
     * */
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        if (this->get_size()[0] > 0) {
            this->comb_->apply(b, x);
        }
    }

    /* Forwarding call of apply which checks if the matrix contains any
     * submatrices
     * */
    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {
        if (this->get_size()[0] > 0) {
            this->comb_->apply(alpha, b, beta, x);
        }
    }

private:
    std::shared_ptr<gko::Combination<scalar>> comb_;
};
