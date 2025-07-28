// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <map>
#include <type_traits>

#include <ginkgo/ginkgo.hpp>

#include "OGL/Solver/PipeCG.hpp"

namespace Foam {

defineTypeNameAndDebug(GKOPipeCG, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<GKOPipeCG>
    addGKOPipeCGSymMatrixConstructorToTable_;
}  // namespace Foam
