# Inter-Frame Compression for Dynamic Point Cloud Geometry Coding

by [Anique Akhtar](https://aniqueakhtar.github.io/), [Zhu Li](http://l.web.umkc.edu/lizhu/), [Geert Van der Auwera](https://www.linkedin.com/in/geertvanderauwera/).

## Introduction

This repository is for our paper '[Inter-Frame Compression for Dynamic Point Cloud Geometry Coding](https://ieeexplore.ieee.org/document/10380494)'. The code uses [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).

Efficient point cloud compression is essential for applications like virtual and mixed reality, autonomous driving, and cultural heritage. This paper proposes a deep learning-based inter-frame encoding scheme for dynamic point cloud geometry compression. We propose a lossy geometry compression scheme that predicts the latent representation of the current frame using the previous frame by employing a novel feature space inter-prediction network. The proposed network utilizes sparse convolutions with hierarchical multiscale 3D feature learning to encode the current frame using the previous frame. The proposed method introduces a novel predictor network for motion compensation in the feature domain to map the latent representation of the previous frame to the coordinates of the current frame to predict the current frame’s feature embedding. The framework transmits the residual of the predicted features and the actual features by compressing them using a learned probabilistic factorized entropy model. At the receiver, the decoder hierarchically reconstructs the current frame by progressively rescaling the feature embedding. The proposed framework is compared to the state-of-the-art Video-based Point Cloud Compression (V-PCC) and Geometry-based Point Cloud Compression (G-PCC) schemes standardized by the Moving Picture Experts Group (MPEG). The proposed method achieves more than 88% BD-Rate (Bjøntegaard Delta Rate) reduction against G-PCCv20 Octree, more than 56% BD-Rate savings against G-PCCv20 Trisoup, more than 62% BD-Rate reduction against V-PCC intra-frame encoding mode, and more than 52% BD-Rate savings against V-PCC P-frame-based inter-frame encoding mode using HEVC. These significant performance gains are cross-checked and verified in the MPEG working group.


## Installation

**We employ G-PCC TMC13 v12 to losslessly encode the coordinates.**
The G-PCC is found in the directory `utils/tmc3`

### Our Environment
- python3.7 or 3.8
- cuda10.2 or 11.0
- pytorch1.6 or 1.7
- MinkowskiEngine 0.5
- torchac~=0.9.3
- tqdm~=4.62.3
- tensorboardX~=2.5
- matplotlib~=3.5.1
- h5py~=3.6.0


### Installation Steps

```
conda create -n py3-mink python=3.8
conda activate py3-mink

conda install openblas-devel -c anaconda
conda install -c anaconda future

pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

```

Alternative installation:
```
conda create -y -n py3-mink python=3.8
conda activate py3-mink

conda install -y cuda -c nvidia/label/cuda-11.3.1
conda install -y cudatoolkit=11.3 -c nvidia
conda install -y pytorch=1.10 torchvision -c pytorch

conda install -y openblas-devel=0.3.2 openblas=0.3.2 ninja -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git ~/MinkowskiEngine
cd ~/MinkowskiEngine
# Optionally checkout a specific tag version here, i.e:
# git checkout -b v0.5.3 tags/v0.5.3
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

Alternative installation:
```
conda create -y -n MY-MINK-ENV-NAME python=3.7
conda activate MY-MINK-ENV-NAME
conda install -y open3d=0.9 -c open3d-admin
conda install -y cuda -c nvidia/label/cuda-11.3.1
conda install -y cudatoolkit=11.3 -c nvidia
conda install -y pytorch=1.10 torchvision -c pytorch
conda install -y openblas-devel=0.3.2 openblas=0.3.2 ninja -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
make clean
# Optionally checkout a specific tag version here.
# i.e. git checkout -b v0.5.4 tags/v0.5.4
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

Alternative Docker Installation for GTX 3090 Ti. Dockerfile:
```
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install keyboard-configuration -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

RUN apt-get --assume-yes install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt install python3.7 -y

#RUN ln -s /usr/bin/python3.8 /usr/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python

RUN apt install python3-pip -y
RUN python -m pip install --upgrade pip
#RUN python -m pip install --upgrade pip3

RUN apt-get install -y\
      build-essential \
      apt-utils \
      ca-certificates \
      wget \
      git \
      vim \
      libssl-dev \
      curl \
      unzip \
      unrar

RUN apt-get install -y libsm6 libxrender1 libfontconfig1 libpython3.7-dev libopenblas-dev
RUN apt install libgl1-mesa-glx -y

RUN python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN python -m pip install matplotlib \
 && python -m pip install pillow \
 && python -m pip install numpy \
 && python -m pip install scipy \
 && python -m pip install cython \
 && python -m pip install scikit-image \
 && python -m pip install sklearn \
 && python -m pip install opencv-python \
 && python -m pip install open3d

RUN TORCH_CUDA_ARCH_LIST="8.6+PTX" python -m pip install -U git+https://github.com/StanfordVL/MinkowskiEngine \
	--install-option="--force_cuda" \
	--install-option="--cuda_home=/usr/local/cuda" \
	--install-option="--blas=openblas"

RUN python -m pip install h5py \
 && python -m pip install tensorboardX \
 && python -m pip install easydict
```


Alternative Docker Installation for GTX 1080 Ti. Dockerfile:
```
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install keyboard-configuration -y

RUN apt install python3.7 -y

RUN ln -s /usr/bin/python3.7 /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python

RUN apt install python3-pip -y
RUN python -m pip install pip

RUN apt-get install -y\
      build-essential \
      apt-utils \
      ca-certificates \
      wget \
      git \
      vim \
      libssl-dev \
      curl \
      unzip \
      unrar

RUN apt-get install -y libsm6 libxrender1 libfontconfig1 libpython3.7-dev libopenblas-dev
RUN apt install libgl1-mesa-glx -y

RUN python -m pip install matplotlib \
 && python -m pip install pillow \
 && python -m pip install numpy \
 && python -m pip install scipy \
 && python -m pip install cython \
 && python -m pip install scikit-image \
 && python -m pip install sklearn \
 && python -m pip install opencv-python \
 && python -m pip install open3d \
 && python -m pip install torch \
 && python -m pip install torchvision

RUN python -m pip install -U git+https://github.com/StanfordVL/MinkowskiEngine \
	--install-option="--force_cuda" \
	--install-option="--cuda_home=/usr/local/cuda" \
	--install-option="--blas=openblas"
```

## Replacement
The MPEG PSNR calculation tool `pc_error_d` and the GPCC `tmc3` files needs to be compiled and replaced in the folder `utils`

## Dataset

The training and evaluation dataset needs to be generated using the following file:
```
cd dataset_creation
python train_dataset.py
python eval_dataset.py
```

### Pretrained Models
The pretrained models are found in the pretrained folder.

### Extra Files
- We provide MPEG's `./utils/pc_error_d` file to be able to calculate the PSNR for the point clouds.


## Citation
Accepted at IEEE Transactions on Image Processing (TIP). 2024.
```
Akhtar, Anique, Zhu Li, and Geert Van der Auwera. "Inter-frame compression for dynamic point cloud geometry coding." IEEE Transactions on Image Processing (2024).
```

### Questions
Please feel free to contact us with any questions. Feel free to open a new issue in this repository for a quick discussion.


## Authors

- Anique Akhtar, and G. Van der Auwera are with Qualcomm Technologies Inc., San Diego, CA, USA.
- Z. Li is with the Department of Computer Science and Electrical Engineering, University of Missouri-Kansas City.
