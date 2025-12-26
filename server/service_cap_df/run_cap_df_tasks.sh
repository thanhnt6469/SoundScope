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
#------------------- This bash script using to run all tasks--------------------

# Get the absolute path of the script directory
PROJECT_DIR=$(realpath "$(dirname "$0")")
INPUT_DIR=$(realpath "${2:-"$PROJECT_DIR/user_input"}")     # Directory containing all input .wav file
OUTPUT_DIR=$(realpath "${3:-"$PROJECT_DIR/user_output"}")   # Directory containing all output of all input .wav file
# SEGMENT_DIR=$(realpath "${4:-"$PROJECT_DIR/s2t_segments"}") # Directory containing temp Whisper segments


# echo $PROJECT_DIR
# echo $INPUT_DIR
# echo $OUTPUT_DIR
# echo $SEGMENT_DIR

# # # ==============Script to run ASC-AED (.env1)
conda deactivate


# # # # =================Script to run deepfake and audio captoning (.env3)
conda activate captioning_deepfake

# -------Deepfake-------
cd "$PROJECT_DIR/model/deepfake/01_feature"
python step01_gen_lfcc_01.py --indir "$INPUT_DIR" --outdir './11_lfcc_01' --delta 'yes'
cd ..
python step02_inf.py  --output_dir "$OUTPUT_DIR"

# # ----------- Audio captioning
cd "$PROJECT_DIR/model/audio_captioning"
python run_audio_captioning.py --project_dir "$PROJECT_DIR" --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"
conda deactivate
