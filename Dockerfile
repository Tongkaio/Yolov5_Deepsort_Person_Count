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
	rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install certifi -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install chardet==4.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install cycler==0.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install dataclasses==0.8 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install easydict -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install idna==3.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install imutils==0.5.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install kiwisolver==1.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install matplotlib==3.3.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install numpy==1.19.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install opencv-python==4.2.0.34 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pyparsing -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install python-dateutil -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install six -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install typing-extensions -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install urllib3 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN ln -s /usr/bin/python3.6 /usr/bin/python
