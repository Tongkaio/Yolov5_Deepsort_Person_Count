# yolov5 deepsort 行人 车辆 跟踪 检测 计数

- 实现了 出/入 分别计数。
- 显示检测类别。
- 默认是 南/北 方向检测，若要检测不同位置和方向，可在 main.py 文件第13行和21行，修改2个polygon的点。
- 默认检测类别：行人、自行车、小汽车、摩托车、公交车、卡车。
- 检测类别可在 detector.py 文件第60行修改。


### 视频

bilibili

[![bilibili](https://github.com/dyh/unbox_yolov5_deepsort_counting/blob/main/cover.jpg?raw=true)](https://www.bilibili.com/video/BV14z4y127XX/ "bilibili")


## 运行环境

- python 3.6+，pip 20+
- pytorch
- pip install -r requirements.txt


## 如何运行

1. 下载代码

    ```
    $ git clone https://github.com/dyh/unbox_yolov5_deepsort_counting.git
    ```
   
   > 因此repo包含weights及mp4等文件，若 git clone 速度慢，可直接下载zip文件：https://github.com/dyh/unbox_yolov5_deepsort_counting/archive/main.zip
   
2. 进入目录

    ```
    $ cd unbox_yolov5_deepsort_counting
    ```

3. 创建 python 虚拟环境

    ```
    $ python3 -m venv venv
    ```

4. 激活虚拟环境

    ```
    $ source venv/bin/activate
    ```
   
5. 升级pip

    ```
    $ python -m pip install --upgrade pip
    ```

6. 安装pytorch

    > 根据你的操作系统、安装工具以及CUDA版本，在 https://pytorch.org/get-started/locally/ 找到对应的安装命令。我的环境是 ubuntu 18.04.5、pip、CUDA 11.0。

    ```
    $ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    ```
   
7. 安装软件包

    ```
    $ pip install -r requirements.txt
    ```

8. 在 main.py 文件中第66行，设置要检测的视频文件路径，默认为 './video/test.mp4'

    > 140MB的测试视频可以在这里下载：https://pan.baidu.com/s/1qHNGGpX1QD6zHyNTqWvg1w 提取码: 8ufq 
 
    ```
    capture = cv2.VideoCapture('./video/test.mp4')
    ```
   
9. 运行程序

    ```
    python main.py
    ```
10. main_modify.py 用于行人过闸机计数
## 使用框架

- https://github.com/Sharpiless/Yolov5-deepsort-inference
- https://github.com/ultralytics/yolov5/
- https://github.com/ZQPei/deep_sort_pytorch

# 构建和运行镜像
构建和运行镜像前先获取 root 权限：
```shell
sudo su
```
## 构建镜像
Dockerfile在Docker文件夹下，进入该文件夹然后：
```shell
docker build -t tongkai2023/yolov5_deepsort:latest
```
或者从 dockerhub 拉取：
```shell
docker pull tongkai2023/yolov5_deepsort:latest
```
## 运行镜像
1、打开x服务器访问控制：
```shell
xhost +
```
2、**创建并运行**容器，需要把下面命令中的`[人员计数代码根目录]`，换成主机里的人员计数代码的根目录，例如：/home/seu/tongkai/yolo_deepsort/unbox_yolov5_deepsort_counting_20230916，注意此路径中不含中文或者空格：
```shell
docker run -it \
--ipc=host \
--env="DISPLAY" \
--gpus=all \
-e PYTHONUNBUFFERED=1 \
-e QT_X11_NO_MITSHM=1 \
-e PYTHONIOENCODING=utf-8 \
--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
-v [人员计数代码根目录]:/home/yolo_deepsort \
-p "8888:8888" \
--rm \
tongkai/yolov5_deepsort:latest \
bash
```
参数说明：
1. 用于显示图形界面的参数：
	- `--ipc=host`：将容器的IPC命名空间设置为与主机共享，这允许容器与主机上的进程进行IPC通信。
	- `--env="DISPLAY"`：设置容器中的图形应用程序将其图形界面显示到主机上的X服务器。
	- `-e QT_X11_NO_MITSHM=1`：用于解决容器向主机发送图形界面时的某些bug。
	- `--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix`创建了一个挂载点，将主机的`/tmp/.X11-unix`目录绑定到容器内的`/tmp/.X11-unix`目录。这是为了允许容器中的图形应用程序与主机上的X11服务器进行通信。
2. 其他参数：
	- `--gpus=all`：允许容器访问所有的GPU资源。这要求主机上已经安装了NVIDIA的GPU驱动和Docker GPU支持。
	- `-e PYTHONUNBUFFERED=1`，解决代码print中文时的某些bug。
	- `-v [人员计数代码根目录]:/home/yolo_deepsort`：将主机的代码目录挂载到容器内的`/home/yolo_deepsort`下。
	- `-p "8888:8888"`：指定容器的端口。
	- `--rm`：此容器退出后会被自动删除。

3、运行代码：
```shell
python main.py
```
