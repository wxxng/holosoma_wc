#!/bin/bash
# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi
source ${SCRIPT_DIR}/source_common.sh
source ${CONDA_ROOT}/bin/activate holomujoco_wc

# Set MuJoCo-specific environment variables
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ROOT}/envs/holomujoco_wc/lib

# Validate environment is properly activated
PYBIN=/home/rllab3/.holosoma_deps/miniconda3/envs/holomujoco_wc/bin/python
if ${PYBIN} -c "import mujoco" 2>/dev/null; then
    echo "MuJoCo environment activated successfully (holosoma_wc)"
    echo "MuJoCo version: $(${PYBIN} -c 'import mujoco; print(mujoco.__version__)')"
    echo "PyTorch version: $(${PYBIN} -c 'import torch; print(torch.__version__)')"
    echo "holosoma source: $(${PYBIN} -c 'import holosoma; print(holosoma.__file__)')"

    # Print mujoco-warp commit if installed
    if ${PYBIN} -c "import mujoco_warp" 2>/dev/null; then
        MUJOCO_WARP_COMMIT=$(git -C ${WORKSPACE_DIR}/mujoco_warp rev-parse --short HEAD 2>/dev/null || echo "unknown")
        echo "MuJoCo Warp commit: ${MUJOCO_WARP_COMMIT}"
    fi
else
    echo "Warning: MuJoCo environment activation may have issues"
fi
