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

# Smoke test: the icoFoam cavity tutorial on 2 ranks, with the pressure solved by OGL's
# GKOCG on the given Ginkgo executor. Fails unless the run completes, every pressure
# solve went through GKOCG, and the last pFinal solve (relTol 0) reached its tolerance.
run_cavity_smoke_test() {
    local executor="$1"
    local case_dir="${PWD}/build/${PRESET}/cavity"
    local tolerance=1e-06
    echo "=== Cavity smoke test with GKOCG on ${executor} ==="

    rm -rf "${case_dir}"
    cp -r "${FOAM_TUTORIALS}/incompressible/icoFoam/cavity/cavity" "${case_dir}"
    # libOGL.so is not installed, OpenFOAM loads it from the build directory
    export LD_LIBRARY_PATH="${PWD}/build/${PRESET}:${LD_LIBRARY_PATH}"
    (
        cd "${case_dir}"
        foamDictionary system/controlDict -entry libs -add '("libOGL.so")' > /dev/null
        foamDictionary system/controlDict -entry endTime -set 0.1 > /dev/null
        # pFinal too: foamDictionary writes pFinal's "$p" expanded, with the old solver
        for field in p pFinal; do
            foamDictionary system/fvSolution -entry "solvers/${field}/solver" -set GKOCG > /dev/null
            foamDictionary system/fvSolution -entry "solvers/${field}/preconditioner" \
                -set none > /dev/null
            foamDictionary system/fvSolution -entry "solvers/${field}/tolerance" \
                -set "${tolerance}" > /dev/null
            foamDictionary system/fvSolution -entry "solvers/${field}/executor" \
                -add "${executor}" > /dev/null
        done
        cat > system/decomposeParDict << 'DICT'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
numberOfSubdomains 2;
method          simple;
coeffs
{
    n           (2 1 1);
}
DICT
        blockMesh > log.blockMesh 2>&1
        decomposePar > log.decomposePar 2>&1
        status=0
        "${MPIEXEC}" -np 2 icoFoam -parallel > log.icoFoam 2>&1 || status=$?

        if [ "${status}" -ne 0 ] || ! grep -q "^End" log.icoFoam; then
            echo "icoFoam failed (exit code ${status}):"
            tail -n 50 log.icoFoam
            exit 1
        fi
        solves=$(grep -c "GKOCG:  Solving for p" log.icoFoam || true)
        others=$(grep "Solving for p," log.icoFoam | grep -vc "GKOCG:" || true)
        last_residual=$(grep "GKOCG:  Solving for p" log.icoFoam | tail -n 1 \
            | sed -n 's/.*Final residual = \([^,]*\),.*/\1/p')
        echo "GKOCG pressure solves: ${solves}, other pressure solves: ${others}," \
            "last final residual: ${last_residual}"
        if [ "${solves}" -eq 0 ] || [ "${others}" -ne 0 ] || [ -z "${last_residual}" ] \
            || ! awk -v r="${last_residual}" -v t="${tolerance}" 'BEGIN { exit !(r <= t) }'; then
            echo "Cavity smoke test failed:"
            grep "Solving for p" log.icoFoam | tail -n 10
            exit 1
        fi
    )
    echo "=== Cavity smoke test passed ==="
}

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
    run_cavity_smoke_test cuda

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
    run_cavity_smoke_test hip

elif [ "$GPU_VENDOR" == "intel" ]; then
    sycl-ls 2>/dev/null | grep '^\[level_zero:gpu\]'

    # Compiler info (non-fatal)
    icpx --version 2>/dev/null | head -1 || echo "icpx not found"

    # Intel PVC has two tiles; COMPOSITE exposes each tile as a separate Level Zero
    # device, so all work and synchronisation stay on one tile.
    export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

    # The libfabric provider Intel MPI picks for the interconnect fails in the container
    # (OFI EP enable failed: Cannot allocate memory). The tests run on one node, so use
    # the TCP provider, next to shared memory within the node.
    export FI_PROVIDER=tcp

    echo "=== Configuring, building, and testing OGL on Intel ==="
    cmake --preset "${PRESET}" \
        -DCMAKE_PREFIX_PATH=/opt/ginkgo \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-sycl-2020-compat" \
        -DMPIEXEC_EXECUTABLE="${MPIEXEC}"
    cmake --build --preset "${PRESET}"
    ctest --preset "${PRESET}" --output-on-failure
    run_cavity_smoke_test sycl

else
    echo "Unknown GPU type: $GPU_VENDOR"
    exit 1
fi
