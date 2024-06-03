FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG DEBIAN_FRONTEND=noninteractive

# Update List of avai. Packages and intall additional packages
RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get -y install \
                        python3-tk \
	&& rm -rf /var/lib/apt/lists/* #cleans up apt cache -> reduces image size
#RUN add-apt-repository universe
RUN apt update
RUN apt-get -y install tzdata \
	htop \
	graphviz \
	sox \
	wget \
	git \
	nano \
	zip \
	libjpeg8-dev \
	zlib1g-dev \
	sudo \
	ffmpeg \
	p7zip-full

RUN rm -rf $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/ruamel*
RUN python3 -m pip install --upgrade -I setuptools
RUN python3 -m pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN python3 -m pip install matplotlib \
	scipy \
	opencv-contrib-python-headless \
	tqdm \
	h5py \
	wandb \
	scikit-image \
	setuptools \
	configargparse \
	segmentation-models-pytorch==0.2

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN adduser user sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER user

VOLUME /ToF_RGB_Fusion
WORKDIR /ToF_RGB_Fusion

#Build with: docker build -t tof_rgb_fusion:1.0 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
