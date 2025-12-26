#!/bin/bash

# Enable error handling
set -e

COMMON_CONDA_PATHS=(
    "/opt/miniconda3/etc/profile.d/conda.sh"
    "/opt/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
)
# Find conda.sh
for conda_path in "${COMMON_CONDA_PATHS[@]}"; do
    if [ -f "$conda_path" ]; then
        CONDA_SH="$conda_path"
        break
    fi
done

# Confirm conda path
if [ -z "$CONDA_SH" ]; then
    echo "conda.sh not found. Please install Miniconda/Anaconda."
    exit 1
fi
source "$CONDA_SH"

# Activate the Conda environment
conda activate captioning_deepfake

# Execute the given command (default is Uvicorn)
exec "$@"