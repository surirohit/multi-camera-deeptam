
# Multi-Camera DeepTAM

Project for the course on 3D Vision at ETH Zurich

## Installation Instructions:

To install the virtual environment and all required dependencies, run:
```bash
./install.sh
```

Source the virtual environment installed:
```bash
workon deeptam_py
```

## Dataset Directory Organization

For each camera, the directory should be organized as shown below:
```bash
data/
    | cam_1/
        | rgb/
        | depth/
        | rgb.txt
        | depth.txt
        | groundtruth.txt
        | config.yaml
    | cam_2/
    ...
```

The text files be similar to the ones _RGBD Freiburg sequences_, i.e. each line should first contain the timestamp information followed by the data:
* __Images__: Data is the file path relative to the `config.yaml` file
* __Groundtruth__: Data is the cartesion poisiotn and quaternion orientation of that particular camera

An example YAML configuration file is present [here](resources/data/sample_config.yaml). Please rename it to `config.yaml` and ensure that the directory for each camera contains this file.
