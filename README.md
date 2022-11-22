# pointpillars

## 简介
这是一个工程环境的demo，集成了两个功能
- 基于[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)特定版本的训练环境，扩增了对数据集的支持，专门用于训练PointPillars模型
- 基于Nvidia官方的[pointpillars的Tensorrt部署工程](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)，对结构做了一些修改，适用于ros推理

## 训练环境介绍

### 环境准备

以下是测试通过环境
| 环境   | OS | GPU | CUDA | CUDNN | Tensorrt |
|---|---|---|---|---|---|
| 环境一 |  Ubuntu20.04 |RTX2080 | 11.3 | 8.2.1 | GA8.0 |
| 环境二 |         |         |      |       |      |

1. 创建conda环境

> Note: 该环境需要使用spconv,其版本需与cuda保持一致，此处cuda为11.3，故使用spconv-cu113

```bash
export CONDA_ENV_NAME=OpenPCDet && \
export PYTHON_VERSION=3.8 && \
export CUDA_VERSION=11.3 && \
export SPCONV_VERSION=113 && \
export TORCH_VERSION=1.11.0 && \
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y && \
conda activate $CONDA_ENV_NAME && \
conda install pytorch=$TORCH_VERSION torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch -y && \
pip3 install spconv-cu$SPCONV_VERSION
```

2. 安装OpenPCDet

> Note 如果出现某些包未找到，那可能需要科学上网

```bash
export CONDA_ENV_NAME=OpenPCDet && \
conda activate $CONDA_ENV_NAME && \
git clone https://github.com/open-mmlab/OpenPCDet.git && \
cd OpenPCDet && \
python setup.py develop
```

3. 安装额外的用于模型转换的库

```bash
export CONDA_ENV_NAME=OpenPCDet && \
conda activate $CONDA_ENV_NAME && \
pip install pyyaml scikit-image onnx onnx-simplifier && \
pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
```
### Train

> Note 
> 
> 不同的数据集格式不同，所以准备数据的时候，格式务必遵循下述的格式准备，此外数据还需要进行转换，转换需要较久的时间，请耐心等待

#### KITTI

**数据格式**
```bash
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib
│   │   │   ├──velodyne
│   │   │   ├──image_2
│   │   │   ├──label_2
│   │   │── testing
│   │   │   ├──calib
│   │   │   ├──velodyne
│   │   │   ├──image_2
```

**数据转换**

```bash
export CONDA_ENV_NAME=OpenPCDet && \
conda activate $CONDA_ENV_NAME && \
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
**train**

```bash
export CONDA_ENV_NAME=OpenPCDet && \
conda activate $CONDA_ENV_NAME && \
cd tools && \
python train.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml
```

#### ONCE

**数据格式**

```bash
OpenPCDet
├── data
│   ├── once
│   │   │── ImageSets
|   |   |   ├──train.txt
|   |   |   ├──val.txt
|   |   |   ├──test.txt
│   │   │── data
│   │   │   ├──000000
|   |   |   |   |──000000.json (infos)
|   |   |   |   |──lidar_roof (point clouds)
|   |   |   |   |   |──frame_timestamp_1.bin
|   |   |   |   |  ...
|   |   |   |   |──cam0[1-9] (images)
|   |   |   |   |   |──frame_timestamp_1.jpg
|   |   |   |   |  ...
|   |   |   |  ...
```

**数据转换**
```bash
On The Way
```

**train**

```bash
On The Way
```

### 模型转换

```bash
export CONDA_ENV_NAME=OpenPCDet && \
conda activate $CONDA_ENV_NAME && \
cd tools && \
python exporter.py --ckpt ./pointpillar_7728.pth
```

## 部署环境介绍

参考[pointpillars_ws](./pointpillars_ws/README.md)，这是一个独立的ros工作空间，用于部署pointpillars模型