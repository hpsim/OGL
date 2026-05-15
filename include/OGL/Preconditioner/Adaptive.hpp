// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <OGL/common.hpp>

#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"

template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class Adaptive : public gko::EnableLinOp<Adaptive<ValueType, IndexType>>,
                 public gko::ConvertibleTo<gko::matrix::Dense<ValueType>>,
                 public gko::WritableToMatrixData<ValueType, IndexType>,
                 public gko::Transposable {
    friend class gko::EnableLinOp<Adaptive>;
    friend class gko::EnablePolymorphicObject<Adaptive, gko::LinOp>;

public:
    using gko::EnableLinOp<Adaptive<ValueType, IndexType>>::convert_to;
    using gko::EnableLinOp<Adaptive<ValueType, IndexType>>::move_to;
    using gko::ConvertibleTo<gko::matrix::Dense<ValueType>>::convert_to;
    using gko::ConvertibleTo<gko::matrix::Dense<ValueType>>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = gko::matrix_data<ValueType, IndexType>;
    using transposed_type = Adaptive<ValueType, IndexType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * @brief `true` means it is known that the matrix given to this
         *        factory will be sorted first by row, then by column index,
         *        `false` means it is unknown or not sorted, so an additional
         *        sorting step will be performed during the preconditioner
         *        generation (it will not change the matrix given).
         *        The matrix must be sorted for this preconditioner to work.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };

    GKO_ENABLE_LIN_OP_FACTORY(Adaptive, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit Adaptive(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<Adaptive>(exec)
    {}

    explicit Adaptive(const Factory *factory,
                      std::shared_ptr<const gko::LinOp> system_matrix)
        : gko::EnableLinOp<Adaptive>(factory->get_executor(),
                                     gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()}
    {
        this->generate(system_matrix.get(), parameters_.skip_sorting);
    }

    void generate(const gko::LinOp *system_matrix, bool skip_sorting) {}

    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override {}

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {}

private:
};
