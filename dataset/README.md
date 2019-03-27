# Dataset Collection

 [AirSim](https://github.com/Microsoft/AirSim) is a simulator by Microsoft for AI research on drones.  This documentation briefly describes the steps involved in collecting the dataset for the project __Multi-Camera DeepTAM__.

The repository contains the required scripts and instructions to collect the dataset using the AirSim simulator.

# Install Dependencies
Python 3.6 is required for running the dataset collection code. Moreover, install the following Python package:

```bash
pip install msgpack-rpc-python
pip install numpy
sudo apt-get install python-opencv
```

# Sensor Setup

The poses collected from the simulator are the position and orientation of the base of the spawned drone in NED format. The transformation for the cameras with respect to the base are given as:
* __Front Left Camera__: TO-DO
* __Front Right Camera__: TO-DO
