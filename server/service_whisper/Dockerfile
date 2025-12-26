FROM my_base_image AS service_whisper

WORKDIR /app

# Accept Conda Terms of Service and create the Conda environment
RUN /opt/conda/bin/conda init bash && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y --name whisper python=3.10.15"

# Set Conda shell settings for activation
SHELL ["/bin/bash", "--login", "-c"]

COPY requirements_whisper.txt .

# Activate Conda in the same RUN command and install dependencies
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate whisper && \
    conda activate whisper && \
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 --timeout 10000 && \
    pip install --no-cache-dir -r requirements_whisper.txt --timeout 1000

RUN apt update && apt install -y ffmpeg

# Create additional directories
RUN mkdir -p /app/user_input /app/user_output /app/model /app/user_final_info /app/s2t_segments 

# # # Copy the .env file into the image temporarily
# COPY .env /app/.env

# # Download cached Hugging Face models 
# RUN source /opt/conda/etc/profile.d/conda.sh && conda activate whisper && \
#     export PYANNOTE_KEY=$(grep -E '^PYANNOTE_KEY=' /app/.env | cut -d '=' -f2- | tr -d '"') && \   
#     python -c "from pyannote.audio import Pipeline; \
#         Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='${PYANNOTE_KEY}');\

#         # from speechbrain.inference.interfaces import foreign_class;\
#         # classifier = foreign_class(source='speechbrain/emotion-recognition-wav2vec2-IEMOCAP', pymodule_file='custom_interface.py', classname='CustomEncoderWav2vec2Classifier');\
#     " && \
#     echo "All cached Huggingface models downloaded successfully" && \


# Copy application files
COPY . .

# Copy the entrypoint script and make it executable - Make sure add this file to container
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]


CMD ["uvicorn", "whisper_based_API:app", "--host", "0.0.0.0", "--port", "8001"]