FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN \
  DEBIAN_FRONTEND=noninteractive apt-get update && \ 
  DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y vim python3 python3-pip && \
  sed -i "s/UsePAM yes/#UsePAM yes/" /etc/ssh/sshd_config && \
  sed -i "s/PermitRootLogin/#PermitRootLogin/" /etc/ssh/sshd_config && \
  echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
  echo 'UsePAM no' >> /etc/ssh/sshd_config && \
  echo 'service ssh start' >> /etc/bash.bashrc && \
  echo 'root:123' | chpasswd  

### pytorch
RUN pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html

### add others packages
RUN pip install pycuda \
    numpy \
    scikit-image \
    tensorboard==2.9.1 \
    pandas\
    opencv-python \
    timm \
    matplotlib \
    scikit-learn \
    setuptools==59.5.0 \