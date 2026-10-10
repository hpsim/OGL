#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
# SPDX-FileCopyrightText: 2026 OGL authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------

set -euo pipefail

# Check required environment variables
GPU_VENDOR=${GPU_VENDOR:?Error: Must set GPU vendor (nvidia|amd|intel)}
PRESET="develop"

echo "Selected GPU type: $GPU_VENDOR"

# The runner sets its own PATH, so the PATH entries of the image are lost: add the
# GPU-aware MPICH of the Ginkgo images and the CUDA toolkit back (missing dirs are harmless).
export PATH="/opt/mpich/bin:/usr/local/cuda/bin:${PATH}"
# The same for the oneAPI compilers and Intel MPI of the SYCL image.
if [ "$GPU_VENDOR" == "intel" ] && [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +eu
    # shellcheck disable=SC1091
    . /opt/intel/oneapi/setvars.sh --force > /dev/null
    set -eu
fi

# The openfoam-ginkgo images (exasim-project/openfoam-conda) have OpenFOAM in /opt/openfoam,
# using the MPI of the image, so OpenFOAM's Pstream and Ginkgo load the same MPI. The image
# activates it through BASH_ENV, which the runner may not pass on; activate.sh is guarded,
# so sourcing it again is harmless.
set +u
# shellcheck disable=SC1091
. /opt/openfoam/activate.sh
set -u
echo "OpenFOAM ${WM_PROJECT_VERSION} (${WM_OPTIONS}, ${FOAM_MPI}) in ${WM_PROJECT_DIR}"

# Launch the MPI tests with the launcher of the MPI OGL links, and start the ranks
# locally: inside the Slurm allocation hydra would otherwise bootstrap through srun,
# and every rank comes up as a singleton.
export HYDRA_BOOTSTRAP=fork
export I_MPI_HYDRA_BOOTSTRAP=fork
MPIEXEC="$(command -v mpiexec.mpich || command -v mpiexec || true)"
echo "=== MPI launcher: ${MPIEXEC} ==="
[ -n "${MPIEXEC}" ] && { "${MPIEXEC}" --version | head -4 || true; }

echo "=== Tool versions ==="
cmake --version
g++ --version || clang++ --version

if [ "$GPU_VENDOR" == "nvidia" ]; then
    echo "=== NVIDIA GPU and compiler driver info ==="
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv
    nvcc --version

    echo "=== Configuring, building, and testing OGL on NVIDIA ==="
    export CUDA_VISIBLE_DEVICES=0
    cmake --preset "${PRESET}" \
        -DCMAKE_PREFIX_PATH=/opt/ginkgo \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DMPIEXEC_EXECUTABLE="${MPIEXEC}"
    cmake --build --preset "${PRESET}"
    ctest --preset "${PRESET}" --output-on-failure

elif [ "$GPU_VENDOR" == "amd" ]; then
    # Set up environment
    CXX_COMPILER_PATH="$(which g++)"
    CXX_SOURCE="${CXX_COMPILER_PATH%/*/*}"
    CXX_LIBDIR="${CXX_SOURCE}/lib64"
    export LD_LIBRARY_PATH=${CXX_LIBDIR}:${LD_LIBRARY_PATH}

    echo "=== AMD GPU and compiler driver info ==="
    rocminfo | grep "AMD"
    hipcc --version

    echo "=== Configuring, building, and testing OGL on AMD ==="
    cmake --preset "${PRESET}" \
        -DCMAKE_PREFIX_PATH="/opt/ginkgo;/opt/rocm" \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_CXX_FLAGS="--gcc-toolchain=${CXX_SOURCE}" \
        -DCMAKE_EXE_LINKER_FLAGS="-L${CXX_LIBDIR}" \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a \
        -DMPIEXEC_EXECUTABLE="${MPIEXEC}"
    cmake --build --preset "${PRESET}"
    ctest --preset "${PRESET}" --output-on-failure

elif [ "$GPU_VENDOR" == "intel" ]; then
    sycl-ls 2>/dev/null | grep '^\[level_zero:gpu\]'

    # Compiler info (non-fatal)
    icpx --version 2>/dev/null | head -1 || echo "icpx not found"

    # Intel PVC has two tiles; COMPOSITE exposes each tile as a separate Level Zero
    # device, so all work and synchronisation stay on one tile.
    export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

    echo "=== Configuring, building, and testing OGL on Intel ==="
    cmake --preset "${PRESET}" \
        -DCMAKE_PREFIX_PATH=/opt/ginkgo \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-sycl-2020-compat" \
        -DMPIEXEC_EXECUTABLE="${MPIEXEC}"
    cmake --build --preset "${PRESET}"
    ctest --preset "${PRESET}" --output-on-failure

else
    echo "Unknown GPU type: $GPU_VENDOR"
    exit 1
fi
