FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      python3.10-venv \
      fonts-dejavu-core \
      rsync \
      git \
      git-lfs \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      libgoogle-perftools4 \
      libtcmalloc-minimal4 \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Set the working directory
WORKDIR /workspace
COPY . .
# RUN git clone https://github.com/ashleykleynhans/runpod-worker-instantid.git

# Install the worker dependencies
WORKDIR /workspace/src
RUN pip3 install --no-cache-dir torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir xformers==0.0.22 runpod && \
    pip3 install -r requirements.txt
    # pip3 install --no-cache-dir huggingface-hub

# Download the checkpoints
RUN python3 download_checkpoints.py

# Download antelopev2 models from Huggingface
RUN git lfs install && \
    git clone https://huggingface.co/ashleykleynhans/FaceAnalysis models

# Docker container start script
COPY --chmod=755 start_standalone.sh /start.sh

# Start the container
ENTRYPOINT /start.sh
