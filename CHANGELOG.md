<!--
SPDX-FileCopyrightText: 2024 OGL authors

SPDX-License-Identifier: GPL-3.0-or-later
-->

# 0.5.4 (unreleased)
- Add a warning message to users who use dpcpp as the executor name [PR #129](https://github.com/hpsim/OGL/pull/129)
- Add support for SP scalars and determine label and scalar size from env variable [PR #120](https://github.com/hpsim/OGL/pull/120) 
- Notify user of a unsupported executor argument [PR #118](https://github.com/hpsim/OGL/pull/118) 
- Notify user of a unsupported executor argument [PR #]
- Add a list of all OGL authors [PR #126](https://github.com/hpsim/OGL/pull/126)
- Replace #ifdef include guard with #pragma once [PR #126](https://github.com/hpsim/OGL/pull/126)
- Switch to SPDX headers, fix formatting issue[PR #122](https://github.com/hpsim/OGL/pull/112)
# 0.5.3 (2024/03/08)
- Fix issue when building against DP labels [PR #111](https://github.com/hpsim/OGL/pull/111)
# 0.5.2 (2024/03/08)
- Fix issue with cyclic boundary conditions [PR #108](https://github.com/hpsim/OGL/pull/108)
# 0.5.1 (2024/01/18)
- Fix issue with failing cmake build 
# 0.5.0 (2024/01/11)
## Features
- Add on device permutation functionality [PR #101](https://github.com/hpsim/OGL/pull/101) 
## Stability and Fixes
- Fix issue with hanging diverging cases [PR #104](https://github.com/hpsim/OGL/pull/96)
- Implemented unit tests for ldu conversion [PR #97](https://github.com/hpsim/OGL/pull/97)
- Move to Ginkgo v1.7.0 [PR #96](https://github.com/hpsim/OGL/pull/96)
- Add integration tests [PR #91](https://github.com/hpsim/OGL/pull/91)
## Performance Data
- Recent single node performance data is available [here](https://github.com/exasim-project/benchmark_data/pull/5)
[![Speedup](https://github.com/exasim-project/benchmark_data/blob/ogl_170_rev1/2024-01-10_09_21/LidDrivenCavity3D/postProcessing/ogl_170_rev1/unprecond_speedup_SolveP_over_nCells_c%3DnProcs_s%3Dsolver_p_cols%3DHost.png)](https://github.com/exasim-project/benchmark_data/blob/ogl_170_rev1/2024-01-10_09_21/LidDrivenCavity3D/postProcessing/ogl_170_rev1/unprecond_speedup_SolveP_over_nCells_c%3DnProcs_s%3Dsolver_p_cols%3DHost.png)
