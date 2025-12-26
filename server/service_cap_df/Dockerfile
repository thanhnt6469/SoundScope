# FROM my_base_image AS service_cap_df

# WORKDIR /app


# # Ensure Conda is initialized and create the Conda environment
# RUN /opt/conda/bin/conda init bash && \
#     /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda create -y --name captioning_deepfake python=3.9.21"

# # Set Conda shell settings for activation
# SHELL ["/bin/bash", "--login", "-c"]

# COPY requirements_captioning_df.txt .

# # Activate Conda in the same RUN command and install dependencies
# RUN source /opt/conda/etc/profile.d/conda.sh && \
#     conda activate captioning_deepfake && \
#     pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 --timeout 10000 && \
#     pip install --no-cache-dir -r requirements_captioning_df.txt --timeout 1000

# # Create additional directories
# RUN mkdir -p /app/user_input /app/user_output /app/model /app/user_final_info /app/s2t_segments 

# # Copy the entrypoint script and make it executable - Make sure add this file to container
# COPY . .

# COPY entrypoint.sh /app/entrypoint.sh

# RUN chmod +x /app/entrypoint.sh

# # Set the entrypoint script
# ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
# CMD ["uvicorn", "cap_df_API:app", "--host", "0.0.0.0", "--port", "8002"]

FROM continuumio/miniconda3:latest AS service_cap_df

WORKDIR /app

# Accept Conda Terms of Service and create Conda env with Python 3.9
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y --name captioning_deepfake python=3.9.21

# Set Conda shell settings for activation
SHELL ["/bin/bash", "--login", "-c"]

# Copy requirements
COPY requirements_captioning_df.txt .

# Activate env and install CPU-only PyTorch + other deps
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate captioning_deepfake && \
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements_captioning_df.txt

# Create working dirs
RUN mkdir -p /app/user_input /app/user_output /app/model /app/user_final_info /app/s2t_segments

# Copy project files
COPY . .

# Entrypoint
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]

CMD ["uvicorn", "cap_df_API:app", "--host", "0.0.0.0", "--port", "8002"]