// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <map>
#include <type_traits>

#include <ginkgo/ginkgo.hpp>

#include "OGL/Solver/CG.hpp"

namespace Foam {

defineTypeNameAndDebug(GKOCG, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<GKOCG>
    addGKOCGSymMatrixConstructorToTable_;
}  // namespace Foam
