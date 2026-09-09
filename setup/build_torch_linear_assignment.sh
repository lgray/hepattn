#!/bin/bash
# Build torch-linear-assignment, the batched Jonker-Volgenant solver behind
# Matcher(device_solver="jv"), against the torch of the current environment.
#
# It is not a pixi dependency on purpose: it is a CUDA extension that has to be compiled with
# nvcc against the exact torch in the environment, for the GPU it will run on, and it has to be
# told to do so -- its setup.py gates CUDA support on torch.cuda.is_available(), which is False
# on a login node, so a plain `pip install` there silently produces a CPU-only build that solves
# GPU costs on the host. Matcher refuses such a build at construction.
#
# Usage, from inside the environment that will run training (the build takes a few minutes):
#
#   pixi run [-e <env>] bash setup/build_torch_linear_assignment.sh [<target dir>]
#
# Environment:
#   TORCH_CUDA_ARCH_LIST  compute capabilities to compile for; default "8.9;10.0" (L4 and B200).
#                         Set it explicitly for other hardware: the build host has no GPU to
#                         detect, and a binary without the right cubin dies at first launch with
#                         "no kernel image is available for execution on the device".
#   CUDA_HOME             nvcc location; defaults to the active environment, which ships CUDA.
#                         Do not point it at a system module the environment cannot see.
#   MAX_JOBS              parallel nvcc jobs, default 4.
#
# The build is left in place in <target dir> (default vendor/torch-linear-assignment under the
# repository, which is gitignored). Rather than pip-installing it into the pixi environment,
# where the next `pixi install` would remove a package the lock file does not know about, put
# the tree on PYTHONPATH when running:
#
#   export PYTHONPATH=<target dir>
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib      # the extension links the environment's libstdc++
#
# then set `device_solver: jv` on the Matcher in the config.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET=${1:-$REPO/vendor/torch-linear-assignment}
UPSTREAM=https://github.com/ivan-chai/torch-linear-assignment.git
COMMIT=9c842e34f29d55c80f4529bf62f520eed1048442 # v0.0.6

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.9;10.0"}
export CUDA_HOME=${CUDA_HOME:-${CONDA_PREFIX:?run this inside the pixi environment, e.g. via pixi run}}
export MAX_JOBS=${MAX_JOBS:-4}

[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "ERROR: no nvcc under CUDA_HOME=$CUDA_HOME" >&2; exit 1; }

if [ ! -d "$TARGET/.git" ]; then
  git clone --quiet "$UPSTREAM" "$TARGET"
fi
git -C "$TARGET" checkout --quiet "$COMMIT"
rm -rf "$TARGET/build"
find "$TARGET" -name '*.so' -delete

echo "Building torch-linear-assignment @ ${COMMIT:0:7} for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
(cd "$TARGET" && python setup.py build_ext --inplace)

# The build must actually contain CUDA support and a cubin for every requested architecture.
PYTHONPATH="$TARGET" python -c "
import torch_linear_assignment._backend as b
assert b.has_cuda(), 'CPU-only build: FORCE_CUDA did not take'
print('has_cuda: True')"
SO=$(find "$TARGET/torch_linear_assignment" -name '_backend*.so' | head -1)
if command -v cuobjdump >/dev/null 2>&1; then
  ELVES=$(cuobjdump --list-elf "$SO")
  for cc in ${TORCH_CUDA_ARCH_LIST//;/ }; do
    arch="sm_${cc/./}"
    case "$ELVES" in *"$arch".cubin*) echo "cubin: $arch" ;; *) echo "ERROR: no $arch cubin in $SO" >&2; exit 1 ;; esac
  done
fi

cat <<MSG

OK: $TARGET

To use it, run with:
  export PYTHONPATH=$TARGET
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
and set device_solver: jv on the Matcher in the config.
MSG
