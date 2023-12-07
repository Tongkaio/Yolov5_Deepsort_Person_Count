FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04
MAINTAINER tongkai<1092019531@qq.com>

ENV MYPATH /home/yolo_deepsort
WORKDIR $MYPATH

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	cmake \
	curl \
	ca-certificates \
	libjpeg-dev \
	libpng-dev \
	libxext-dev \
	libxrender1 \
	libsm6 \
	libglib2.0-dev \
	locales \
	python3.6 \
	python3.6-dev \
	python3-pip && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools

RUN pip install --no-cache-dir torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt /tmp/requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r /tmp/requirements.txt \
	&& rm /tmp/requirements.txt

RUN ln -s /usr/bin/python3.6 /usr/bin/python