// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <ginkgo/ginkgo.hpp>

class PreconditionerWrapper {

    /* @brief instantiate a local preconditioner and return LinOp Ptr */
    virtual std::shared_ptr<gko::LinOp> create_local();

    /* @brief instantiate a local preconditioner and return LinOp Ptr */
    virtual std::shared_ptr<gko::LinOp> create_multilevel();
};
