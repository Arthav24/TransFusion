ARG PYTORCH="2.7.0"
ARG CUDA="12.8"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


RUN apt-get update && apt-get install -y python3-pip vim git curl wget ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
WORKDIR /workspace
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
RUN git clone https://github.com/michigan-traffic-lab/nuCarla.git
RUN cd /workspace/nuCarla/BEVFormer/mmcv && MMCV_WITH_OPS=1 pip install . --no-cache-dir -v

WORKDIR /workspace/nuCarla/BEVFormer
RUN pip install --no-cache-dir -e .


CMD ["bash"]
