
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
    ├── depth
    ├── depth.txt
    ├── groundtruth.txt
    ├── rgb
    └── rgb.txt
```

The text files be similar to the ones present in [_TUM RGBD sequences_](https://vision.in.tum.de/data/datasets/rgbd-dataset), i.e. each line should first contain the timestamp information followed by the data:
* __Images__: Data is the file path relative to the sequence directory name specified in the `config.yaml` file
* __Groundtruth__: Data is the cartesian position and quaternion orientation of that particular camera (in world/camera frame)

An example YAML configuration file for the RGBD Freiburg1 Desk Sequence is present [here](resources/hyperparameters/freiburg1_config.yaml). 

__NOTE:__ Please ensure that the sequence directory and camera intrinsics are correctly modified according to the dataset. 

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
    --config_file ../resources/hyperparameters/freiburg1_config.yaml
    --weights ../resources/weights/deeptam_tracker_weights/snapshot-300000

```
