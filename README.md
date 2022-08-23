#  CenterPoint inferenced with TensorRT

This repository contains sources and model for [CenterPoint](https://arxiv.org/abs/2006.11275) inference using TensorRT.
The model is created with [mmdetection3d](https://github.com/Tartisan/mmdetection3d).

Overall inference has five phases:

- Convert points cloud into 4-channle voxels
- Extend 4-channel voxels to 10-channel voxel features
- Run pfe TensorRT engine to get 64-channel voxel features
- Run rpn backbone TensorRT engine to get 3D-detection raw data
- Parse bounding box, class type and direction

## Model && Data

The demo used the custom dataset like KITTI.
The onnx file can be converted by [onnx_tools](https://github.com/Tartisan/mmdetection3d/tree/master/tools/onnx_tools/centerpoint)

### Prerequisites

To build the centerpoint inference, **TensorRT** and **CUDA** are needed.

## Environments

- NVIDIA RTX A4000 Laptop GPU
- CUDA 11.1 + cuDNN 8.2.1 + TensorRT 8.2.3

### Compile && Run

```shell
$ mkdir build && cd build
$ cmake .. && make -j$(nproc)
$ ./demo
```

### Visualization

You should install `open3d` in python environment.

```shell
$ cd tools
$ python viewer.py
```

| trt fp16 | pytorch |
| -------- | ------- |
| ![trt fp16](https://tva3.sinaimg.cn/large/0080fUsgly1h5gqbu1ie8j31c10qak9o.jpg) | ![pytorch](https://tvax2.sinaimg.cn/large/0080fUsgly1h532pas0xmj31ey0rlask.jpg) |


#### Performance in FP16

```
| Function(unit:ms) | NVIDIA RTX A4000 Laptop GPU |
| ----------------- | --------------------------- |
| Preprocess        | 0.950476 ms                 |
| Pfe               | 4.37507  ms                 |
| Scatter           | 0.204093 ms                 |
| Backbone          | 9.84435  ms                 |
| Postprocess       | 2.91952  ms                 |
| Summary           | 18.2961  ms                 |
```

## Note

- The pretrained model in this project doesn't predict vx, vy.
- The demo will cache the onnx file to improve performance. If a new onnx will be used, please remove the cache file in "./model".

## References

- [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275)
- [mmdetection3d](https://github.com/Tartisan/mmdetection3d)
- [tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Abraham423/CenterPoint](https://github.com/Abraham423/CenterPoint)
