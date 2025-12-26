#!/bin/bash
COMMON_CONDA_PATHS=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
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
#------------------- This bash script using to run ASC-AED tasks--------------------

# Get the absolute path of the script directory
PROJECT_DIR=$(realpath "$(dirname "$0")")
INPUT_DIR=$(realpath "${2:-"$PROJECT_DIR/user_input"}")     # Directory containing all input .wav file
OUTPUT_DIR=$(realpath "${3:-"$PROJECT_DIR/user_output"}")   # Directory containing all output of all input .wav file


# # # ==============Script to run ASC-AED service (.env1)
conda deactivate

conda activate asc_aed
cd "$PROJECT_DIR/model/asc_aed"
python run_asc_aed.py --project_dir "$PROJECT_DIR" --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"


