
# Multi-Camera DeepTAM

Project for the course on 3D Vision at ETH Zurich

This code has been tested on a computer with following specifications:
* __OS Platform and Distribution:__ Linux Ubuntu 16.04LTS
* __CUDA/cuDNN version:__ CUDA 9.0.176, cuDNN 7.1.4
* __GPU model and memory:__ NVidia GeForce GTX 1070-MaxQ, 8GB
* __Python__: 3.5.2
* __TensorFlow:__ 1.9.0

## Dataset Directory Organization

For each camera, the directory should be organized as shown below:
```bash
data/
└── cam_1/  
    ├── config.yaml
    ├── depth
    ├── depth.txt
    ├── groundtruth.txt
    ├── rgb
    └── rgb.txt
```

The text files be similar to the ones [_TUM RGBD sequences_](https://vision.in.tum.de/data/datasets/rgbd-dataset), i.e. each line should first contain the timestamp information followed by the data:
* __Images__: Data is the file path relative to the `config.yaml` file
* __Groundtruth__: Data is the cartesion position and quaternion orientation of that particular camera (in world/camera frame)

An example YAML configuration file is present [here](resources/data/sample_config.yaml). Please rename it to `config.yaml` and ensure that the directory for each camera contains this file.
```bash
cp resources/data/sample_config.yaml /path/to/dataset/config.yaml
```

## Installation Instructions:

To install the virtual environment and all required dependencies, run:
```bash
./install.sh
```

Source the virtual environment installed:
```bash
workon deeptam_py
```

## Examples:

### Single Camera DeepTAM

To run DeepTAM with a single camera setup, run:
```bash
cd scripts
# run the python script
python single_camera_tracking.py \
    --data_dir ../resources/data/rgbd_dataset_freiburg1_desk \
    --weights ../resources/weights/deeptam_tracker_weights/snapshot-300000
```
