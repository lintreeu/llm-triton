# 使用 Ubuntu 22.04 + CUDA 12.8 的官方開發 image
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
ARG WORKSPACE_DIR=/workspace

# 安裝基本開發工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl ca-certificates \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 設定 python3 為預設 python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 升級 pip
RUN pip3 install --upgrade pip

# 安裝 PyTorch（CUDA 11.3/11.5相容版）
# RUN pip3 install \
#     torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
#     --extra-index-url https://download.pytorch.org/whl/cu113

# 安裝 Triton
# 下載並安裝 Triton
RUN pip install triton==3.3.0

# 安裝其他pip套件
RUN pip install sentencepiece>=0.1.99 numpy>=1.26

# 安裝 JupyterLab + ipykernel（方便 notebook 開發）
RUN pip3 install jupyterlab ipykernel

# 建立一個專用的 kernel 讓 Jupyter 使用
RUN python3 -m ipykernel install --user --name python-triton --display-name "Python (Triton CUDA 11.5)"

# 設定工作目錄
WORKDIR ${WORKSPACE_DIR}

# 開啟 Jupyter port
EXPOSE 8888

# 預設啟動 Jupyter Lab
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
# 啟用bash
CMD ["bash"]
