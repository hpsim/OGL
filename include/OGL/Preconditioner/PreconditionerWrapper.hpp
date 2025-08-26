// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <ginkgo/ginkgo.hpp>

namespace Foam {

class PreconditionerWrapper {
protected:
    std::shared_ptr<gko::Executor> exec_;

    std::shared_ptr<const gko::LinOp> mtx_;

public:
    PreconditionerWrapper(std::shared_ptr<gko::Executor> exec,
                          std::shared_ptr<const gko::LinOp> mtx)
        : exec_(exec), mtx_(mtx)
    {}

    /* @brief instantiate a local preconditioner and return LinOp Ptr */
    virtual std::shared_ptr<gko::LinOp> create();

    /* @brief instantiate a local preconditioner and return LinOp Ptr */
    // virtual std::shared_ptr<gko::LinOp> create_local();

    // /* @brief instantiate a local preconditioner and return LinOp Ptr */
    // virtual std::shared_ptr<gko::LinOp> create_multilevel();
};

}  // namespace Foam
