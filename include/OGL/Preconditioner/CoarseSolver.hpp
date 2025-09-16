// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"

std::shared_ptr<const gko::LinOpFactory> generate_coarse_solver(
    std::shared_ptr<gko::Executor> exec, const dictionary &d, label verbose);
