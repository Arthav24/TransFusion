# Bot-talion changes
- create container of image using
``docker run -it --name TransFusion_train --network=host --shm-size=32g -v /home/arthavpc/703_project:/home/project -v /media/arthavpc/exSSD:/home/data -v /media/arthavpc/ExternalData:/home/externalData --workdir=/home/project --gpus=all --runtime=nvidia --env DISPLAY=$DISPLAY transfusion:1rc6 bash``


- download mini dataset from nuscenes to verify data loading https://www.nuscenes.org/download#
- ``python3 tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini``
- had to reinstall mmcv as it was giving no kernel for cuda:0 error
- pip3 uninstall mmcv-full
- cd /workspace/nuCarla/BEVFormer/mmcv
- rm -rf build mmcv_full.egg-info
- export TORCH_CUDA_ARCH_LIST=12.0 && export CUDA_HOME=/usr/local/cuda
- MMCV_CUDA_ARGS="-gencode=arch=compute_120,code=sm_120" MMCV_WITH_OPS=1 pip install . --no-cache-dir -v
- the gencode arch will change 120 is for RTX 50x0
- run python3 misc/check_mmcv_installation.py for verification
- can also run python -c 'import mmcv; import mmcv.ops' to verify no errors in import
- uninstall mmdet3d  pip3 uninstall mmdet3d && rm -rf ./build
- MMDET3D_CUDA_ARGS="-gencode=arch=compute_120,code=sm_120" pip install -e . --no-cache-dir -v
- pip3 uninstall spconv cumm (by default install cpu version)
- pip3 install 'cumm-cu128'
- pip3 install 'spconv-cu126'

Now the main problem is with two package networkx and trimesh
I had to manually change np.bool to bool using ``find . -name *.py" -exec sed -i 's/np.bool/bool/g' {} +`` in dir /opt/conda/lib/python3.11/site-packages/trimesh
cp misc/graphml.py to /opt/conda/lib/python3.11/site-packages/networkx/readwrite

Its quite hacky because of numpy version and changes will resolve later
after doing all this you should be able to ``python3 demo/pcd_demo.py demo/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth --show``





Python script to convert Lidar model with ResNet

import torch
pts = torch.load('/home/project/TransFusion/work_dirs/transfusion_nusc_pillar_L/epoch_20.pth',  map_location='cpu')

img = torch.load('/home/project/TransFusion/checkpoints/resnet50.pth',  map_location='cpu')

def extract_state(d):
    # If checkpoint already contains a 'state_dict' key
    if "state_dict" in d:
        return d["state_dict"]

    # If checkpoint contains only weights (no wrapper)
    if all(isinstance(k, str) for k in d.keys()):
        return d

    # Try common alternative keys
    for key in ["model", "weights", "state", "params"]:
        if key in d:
            return d[key]

    raise KeyError("No state_dict found in checkpoint. Keys: " + str(d.keys()))

img_state = extract_state(img)
pts_state = extract_state(pts)

new_model = {"state_dict": pts_state.copy()}

for k, v in img_state.items():
    if 'backbone' in k or 'neck' in k:
        new_model["state_dict"]['img_'+k] = v

torch.save(new_model, "fusion_model.pth")


# TransFusion repository

PyTorch implementation of TransFusion for CVPR'2022 paper ["TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers"](https://arxiv.org/abs/2203.11496), by Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu and Chiew-Lan Tai.

This paper focus on LiDAR-camera fusion for 3D object detection. If you find this project useful, please cite:

```bash
@article{bai2021pointdsc,
  title={{TransFusion}: {R}obust {L}iDAR-{C}amera {F}usion for {3}D {O}bject {D}etection with {T}ransformers},
  author={Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu and Chiew-Lan Tai},
  journal={CVPR},
  year={2022}
}
```

## Introduction

LiDAR and camera are two important sensors for 3D object detection in autonomous driving. Despite the increasing popularity of sensor fusion in this field, the robustness against inferior image conditions, e.g., bad illumination and sensor misalignment, is under-explored. Existing fusion methods are easily affected by such conditions, mainly due to a hard association of LiDAR points and image pixels, established by calibration matrices.
We propose TransFusion, a robust solution to LiDAR-camera fusion with a soft-association mechanism to handle inferior image conditions. Specifically, our TransFusion consists of convolutional backbones and a detection head based on a transformer decoder. The first layer of the decoder predicts initial bounding boxes from a LiDAR point cloud using a sparse set of object queries, and its second decoder layer adaptively fuses the object queries with useful image features, leveraging both spatial and contextual relationships. The attention mechanism of the transformer enables our model to adaptively determine where and what information should be taken from the image, leading to a robust and effective fusion strategy. We additionally design an image-guided query initialization strategy to deal with objects that are difficult to detect in point clouds. TransFusion achieves state-of-the-art performance on large-scale datasets. We provide extensive experiments to demonstrate its robustness against degenerated image quality and calibration errors. We also extend the proposed method to the 3D tracking task and achieve the 1st place in the leaderboard of nuScenes tracking, showing its effectiveness and generalization capability.

![pipeline](resources/pipeline.png)

**updates**
- March 23, 2022: paper link added
- March 15, 2022: initial release

## Main Results

Detailed results can be found in [nuscenes.md](configs/nuscenes.md) and [waymo.md](configs/waymo.md). Configuration files and guidance to reproduce these results are all included in [configs](configs), we are not going to release the pretrained models due to the policy of Huawei IAS BU.

### nuScenes detection test

| Model   | Backbone | mAP | NDS  | Link  |
|---------|--------|--------|---------|---------|
| [TransFusion-L](configs/transfusion_nusc_voxel_L.py) | VoxelNet | 65.52 | 70.23 | [Detection](https://drive.google.com/file/d/1Wk8p2LJEhwfKfhsKzlU9vDBOd0zn38dN/view?usp=sharing)
| [TransFusion](configs/transfusion_nusc_voxel_LC.py) | VoxelNet | 68.90 | 71.68 | [Detection](https://drive.google.com/file/d/1X7_ig4v5A2vKsiHtUGtgeMN-0RJKsM6W/view?usp=sharing)

### nuScenes tracking test

| Model | Backbone | AMOTA |  AMOTP   | Link  |
|---------|--------|--------|---------|---------|
| [TransFusion-L](configs/transfusion_nusc_voxel_L.py) | VoxelNet | 0.686 | 0.529 | [Detection](https://drive.google.com/file/d/1Wk8p2LJEhwfKfhsKzlU9vDBOd0zn38dN/view?usp=sharing) / [Tracking](https://drive.google.com/file/d/1pKvRBUsM9h1Xgturd0Ae_bnGt0m_j3hk/view?usp=sharing)|
| [TransFusion](configs/transfusion_nusc_voxel_LC.py)| VoxelNet | 0.718 | 0.551 | [Detection](https://drive.google.com/file/d/1X7_ig4v5A2vKsiHtUGtgeMN-0RJKsM6W/view?usp=sharing) / [Tracking](https://drive.google.com/file/d/1EVuS-MAg_HSXUVqMrXEs4-RpZp0p5cfv/view?usp=sharing)|

### waymo detection validation

| Model   | Backbone | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   |
|---------|--------|---------|---------|---------|---------|
| [TransFusion-L](configs/transfusion_waymo_voxel_L.py) | VoxelNet | 65.07 | 63.70 | 65.97 | 64.91
| [TransFusion](configs/transfusion_waymo_voxel_LC.py) | VoxelNet | 65.11 | 64.02 | 67.40 | 65.51

## Use TransFusion

**Installation**

Please refer to [getting_started.md](docs/getting_started.md) for installation of mmdet3d. We use mmdet 2.10.0 and mmcv 1.2.4 for this project.

**Benchmark Evaluation and Training**

Please refer to [data_preparation.md](docs/data_preparation.md) to prepare the data. Then follow the instruction there to train our model. All detection configurations are included in [configs](configs/).

Note that if you a the newer version of mmdet3d to prepare the meta file for nuScenes and then train/eval the TransFusion, it will have a wrong mAOE and mASE because mmdet3d has a [coordinate system refactoring](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/compatibility.md#coordinate-system-refactoring) which affect the definitation of yaw angle and object size (`l, w`).

## Acknowlegement

We sincerely thank the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [CenterPoint](https://github.com/tianweiy/CenterPoint), [GroupFree3D](https://github.com/zeliu98/Group-Free-3D) for open sourcing their methods.
