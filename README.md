# Team06-ADS_Autonomous-Lane-Detection

instalar o NumPy, PyCUDA e OpenCV com suporte a CUDA no Jetson Nano. O Jetson Nano usa uma versão do Ubuntu (geralmente 18.04 com JetPack) e possui uma GPU NVIDIA, então precisamos garantir que as instalações sejam compatíveis com a arquitetura aarch64 e o ambiente CUDA. Vou assumir que você tem o JetPack instalado (recomendo a versão 4.6.x para suporte estável) e está usando Python 3.6 ou 3.8, que são comuns no Jetson Nano.

### Pré-requisitos

1. Atualize o sistema:
sudo apt-get update && sudo apt-get upgrade -y
2. Verifique o ambiente CUDA: O JetPack já inclui CUDA e cuDNN. Confirme com:
nvcc --version
Você deve ver algo como CUDA 10.2. Se não, instale o JetPack via NVIDIA SDK Manager.
3. Instale ferramentas básicas:
    sudo apt-get install -y build-essential python3-dev python3-pip git

## 1. Instalando o NumPy

O NumPy geralmente já vem com o JetPack, mas para garantir uma versão compatível (ex.: 1.19.4, recomendada para evitar conflitos com PyCUDA e TensorFlow):
    pip3 install numpy==1.19.4

### Verifique a instalação:
    python3 -c "import numpy; print(numpy.__version__)"

### Se houver conflitos, desinstale e reinstale:
    pip3 uninstall numpy
    pip3 install numpy==1.19.4

## 2. Instalando o PyCUDA

O PyCUDA permite que Python acesse a GPU via CUDA. Ele não está no repositório padrão do pip para aarch64, então usaremos rodas pré-compiladas do projeto jetson-nano-wheels.

### 1. Instale dependências:
    sudo apt-get install -y libopenblas-base libopenmpi-dev
### 2. Instale NumPy e pyopencl (dependências do PyCUDA):
    pip3 install 'https://github.com/jetson-nano-wheels/python3.6-numpy-1.19.4/releases/download/v0.0.1/numpy-1.19.4-cp36-cp36m-linux_aarch64.whl'
    pip3 install 'https://github.com/jetson-nano-wheels/python3.6-pyopencl-2021.2.6/releases/download/v0.0.1/pyopencl-2021.2.6-cp36-cp36m-linux_aarch64.whl'
### 3. Instale o PyCUDA:
    pip3 install 'https://github.com/jetson-nano-wheels/python3.6-pycuda-2021.1/releases/download/v0.0.1/pycuda-2021.1-cp36-cp36m-linux_aarch64.whl'
### 4. Corrija possíveis erros de cabeçalho: Se houver erro de xlocale.h, crie um link simbólico:
    sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
### 5. Teste o PyCUDA:
    python3 -c "import pycuda.driver as cuda; cuda.init(); print(cuda.Device(0).name())"
    Deve mostrar algo como "NVIDIA Tegra X1".

## 3. Instalando o OpenCV com suporte a CUDA
O JetPack inclui uma versão do OpenCV, mas sem suporte a CUDA. Para usar a GPU, precisamos compilar o OpenCV do zero. Este processo é longo (2-3 horas) e requer memória swap adicional.

### 3. Aumente o espaço de swap: O Jetson Nano tem 4 GB de RAM + 2 GB de swap, insuficiente para compilar. Aumente o swap:
    sudo systemctl stop dphys-swapfile
    sudo systemctl disable dphys-swapfile
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
Verifique:
    free -m
Deve mostrar ~6-8 GB de swap total.
### 2. Instale dependências:
    sudo apt-get install -y build-essential cmake git unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libgtk2.0-dev libcanberra-gtk* libxvidcore-dev libx264-dev libgtk-3-dev \
    libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev v4l-utils \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libavresample-dev libvorbis-dev libxine2-dev libfaac-dev libmp3lame-dev \
    libtheora-dev libopencore-amrnb-dev libopencore-amrwb-dev \
    libopenblas-dev libatlas-base-dev libblas-dev liblapack-dev \
    libeigen3-dev gfortran libhdf5-dev protobuf-compiler \
    libprotobuf-dev libgoogle-glog-dev libgflags-dev
### 3. Baixe o OpenCV e OpenCV_contrib: Vamos usar a versão 4.5.1, que é estável no Jetson Nano:
    cd ~
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.1.zip
    unzip opencv.zip
    unzip opencv_contrib.zip
    mv opencv-4.5.1 opencv
    mv opencv_contrib-4.5.1 opencv_contrib
    rm opencv.zip opencv_contrib.zip
### 4. Configure o build com CMake:
    cd ~/opencv
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
       -D CMAKE_INSTALL_PREFIX=/usr/local \
       -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
       -D WITH_CUDA=ON \
       -D CUDA_ARCH_BIN=5.3 \
       -D CUDA_ARCH_PTX="" \
       -D WITH_CUDNN=ON \
       -D WITH_CUBLAS=ON \
       -D ENABLE_FAST_MATH=ON \
       -D CUDA_FAST_MATH=ON \
       -D OPENCV_DNN_CUDA=ON \
       -D ENABLE_NEON=ON \
       -D WITH_QT=OFF \
       -D WITH_OPENMP=ON \
       -D WITH_OPENGL=ON \
       -D BUILD_TIFF=ON \
       -D WITH_FFMPEG=ON \
       -D WITH_GSTREAMER=ON \
       -D WITH_TBB=ON \
       -D BUILD_TBB=ON \
       -D BUILD_TESTS=OFF \
       -D WITH_EIGEN=ON \
       -D WITH_V4L=ON \
       -D WITH_LIBV4L=ON \
       -D OPENCV_ENABLE_NONFREE=ON \
       -D INSTALL_C_EXAMPLES=OFF \
       -D INSTALL_PYTHON_EXAMPLES=OFF \
       -D BUILD_NEW_PYTHON_SUPPORT=ON \
       -D BUILD_opencv_python3=ON \
       -D OPENCV_GENERATE_PKGCONFIG=ON \
       -D BUILD_EXAMPLES=OFF ..
### 5. Compile e instale:
    make -j4
    sudo make install
    sudo ldconfig
Nota: Mova o cursor do mouse ocasionalmente para evitar que o screensaver interrompa a compilação. Se falhar perto de 100%, repita make -j4.
### 6.Configure o PYTHONPATH: Adicione o caminho do OpenCV ao Python:
    echo "export PYTHONPATH=/usr/local/lib/python3.6/site-packages:$PYTHONPATH" >> ~/.bashrc
    source ~/.bashrc
Ajuste python3.6 para sua versão do Python, se necessário (use python3 --version).
### 7. Teste o OpenCV com CUDA:
    python3 -c "import cv2; print(cv2.__version__); print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"
Deve mostrar a versão (ex.: 4.5.1) e CUDA: 1 se a GPU estiver habilitada.
### 8. Restaure o swap: Após a compilação, desative o swap extra:
    sudo swapoff /swapfile
    sudo rm /swapfile
    sudo systemctl enable dphys-swapfile
    sudo systemctl start dphys-swapfile

Dicas e Solução de Problemas

  - Erro de memória: Se a compilação do OpenCV falhar, aumente o swap para 8 GB temporariamente.
  - PyCUDA não importa: Verifique se o CUDA está funcional (nvcc --version) e reinstale as rodas do PyCUDA.
  - OpenCV sem CUDA: Confirme que -D WITH_CUDA=ON e -D CUDA_ARCH_BIN=5.3 estão no CMake. Verifique a saída do CMake para garantir que os módulos CUDA estão habilitados.
  - Monitoramento: Use jtop para monitorar a GPU durante os testes:
      sudo pip3 install jetson-stats
      sudo systemctl restart jetson_stats
      jtop

Teste Final

Crie um script Python para verificar as três bibliotecas:
```python
import numpy as np
import pycuda.driver as cuda
import cv2

print("NumPy version:", np.__version__)
cuda.init()
print("PyCUDA - GPU:", cuda.Device(0).name())
print("OpenCV version:", cv2.__version__)
print("CUDA support:", cv2.cuda.getCudaEnabledDeviceCount() > 0)
```


## line Dect

### Train YOLOv5 with dataset

python train.py --img 320 --batch 16 --epochs 30 --data Stop_data/data.yaml --weights yolov5n.pt --device 0 --cache

### onnx -> tensorrt on nano
/usr/src/tensorrt/bin/trtexec --onnx=unet_model_256.onnx --saveEngine=unet_model_256.engine --fp16


### Dataset Roboflow
#### Install
$ pip install roboflow

#### Authenticate
$ roboflow login

#### Import
        roboflow import -w objetctdetectionseame -p objectdetection_seame /path/to/data
                    



## Running YoloV5 with TensorRT Engine on Jetson

python3 -m venv yolov5_env
source yolov5_env/bin/activate

1. 
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

2. 
python train.py \
  --img 320 \
  --batch 8 \
  --epochs 50 \
  --data coco128.yaml \
  --weights yolov5n.pt \
  --device 0 \
  --workers 4

3. Val
   python val.py --data /content/yolov5/dataset/data.yaml --weights /content/yolov5/runs/train/exp/weights/best.pt

4. test
python detect.py --source Stop_data/test/images --weights runs/train/exp/weights/best.pt --img 320 --conf 0.4

5. pth -> onnx
python export.py \
  --weights runs/train/crosswalk-pedestrian7/weights/best.pt \
  --include onnx \
  --opset 13 \
  --imgsz 320

6. (jetson nano) onnx -> tensorrt(engine)
/usr/src/tensorrt/bin/trtexec --onnx=yolov5_crosswalk.onnx --saveEngine=yolov5_crosswalk.engine --workspace=2048 --fp16


## Running YoloV8 with TensorRT Engine on Jetson
1. Install YOLOv8 (Ultralytics)

pip install ultralytics

2. Train with YOLOv8

yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=100 imgsz=320 batch=16

opcional - "device=0"

3. 

yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=dataset.yaml

4. 

yolo export model=runs/detect/train/weights/best.pt format=engine





## Support material
 ```bash
   https://github.com/mailrocketsystems/JetsonYolov5/blob/main/README.md

   https://www.youtube.com/watch?v=ErWC3nBuV6k&list=PLWw98q-Xe7iGIwrnBY_SpXHZzAZZ6944l

   https://www.youtube.com/watch?v=bcM5AQSAzUY&t=199s
```


python3 -m venv yolov5_env

source yolov5_env/bin/activate

1. 
git clone https://github.com/ultralytics/yolov8
cd yolov5
pip install -r requirements.txt

2. 
python train.py --img 320 --batch 16 --epochs 50 --data coco128.yaml --weights yolov5n.pt --name yolov5_coco128

python train.py \
  --img 320 \
  --batch 8 \
  --epochs 50 \
  --data coco128.yaml \
  --weights yolov5n.pt \
  --device 0 \
  --workers 4

3. python export.py --weights runs/train/yolov5n_coco128/weights/best.pt --include onnx --opset 13 --simplify

python export.py \
  --weights runs/train/exp/weights/best.pt \
  --include onnx \
  --opset 13 \
  --imgsz 320

4. (jetson nano)
/usr/src/tensorrt/bin/trtexec --onnx=best_320.onnx --saveEngine=best_320.engine --fp16


