FROM my_base_image AS service_asc_aed

WORKDIR /app

# Accept Conda Terms of Service and create the Conda environment
RUN /opt/conda/bin/conda init bash && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y --name asc_aed python=3.12.4"
    

SHELL ["/bin/bash", "--login", "-c"]

# Install dependencies
COPY requirements_asc_aed.txt .

# RUN source /opt/conda/etc/profile.d/conda.sh && \
#     conda activate asc_aed && \
#     pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 --timeout 10000 && \
#     pip install --no-cache-dir -r requirements_asc_aed.txt --timeout 1000

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate asc_aed && \
    pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 --timeout 1000 && \
    pip install --no-cache-dir -r requirements_asc_aed.txt

# Create additional directories
RUN mkdir -p /app/user_input /app/user_output /app/model 

COPY . .

# Copy the entrypoint script and make it executable - Make sure add this file to container
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]

CMD ["uvicorn", "asc_aed_API:app", "--host", "0.0.0.0", "--port", "8000"]