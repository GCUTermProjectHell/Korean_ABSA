FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
WORKDIR /workspace
COPY . .
# Install Dependencies
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
# Install Dependencies of Container
RUN apt-get update --fix-missing && \
    apt-get install -y wget git
# Git Clone    
RUN git clone https://github.com/GCUTermProjectHell/Korean_ABSA
# Package Install
RUN pip install -r requirements.txt

CMD [ "/bin/bash" ]
