# Dataset Collection

This documentation briefly describes the steps involved in collecting the dataset for the project __Multi-Camera DeepTAM__.

---

# AirSim

[AirSim](https://github.com/Microsoft/AirSim) is a simulator by Microsoft for AI research on drones.  

## Install Dependencies

* You would need to install Unreal Engine and AirSim before you may start working. To do so, follow the instructions [here](https://microsoft.github.io/AirSim/docs/build_linux/).

* Copy the file `settings.json` ([link](airsim/settings.json)) to `~/Documents/AirSim`

* Further, Python 3.5 is required for running the dataset collection code. Moreover, install the following Python package:

```bash
pip install msgpack-rpc-python
pip install numpy
pip install airsim
sudo apt-get install python-opencv
```

## Sensor Setup

The poses collected from the simulator are the position and orientation of the base of the spawned drone in NED format. The transformation for the cameras with respect to the base are given as:
* __Front Left Camera__: TO-DO
* __Front Right Camera__: TO-DO

## Running the program

For the sake of sanity, run the scripts in the following order parallely:
```bash
# Terminal 1:
# script to start dataset collection
python airsim/airsim_collection.py

# Terminal 2:
# script to fly the drone
python airsim/airsim_fly.py
```

---

# MINOS

[MINOS](https://github.com/minosworld/minos) is a simulator designed to support the development of multisensory models for goal-directed navigation in complex indoor environments.

## Install Dependencies

* Clone the following repository from GitHub and follow installation instructions for MINOS available [here](https://github.com/minosworld/minos#installing):
```bash
cd ~/git
git clone https://github.com/minosworld/minos.git
```

* Download the SUNCG dataset using the script ([link](suncg/download_suncg.py))
```
python suncg/download_suncg.py -v 2
```

* Copy the following files from the directory [minos](minos):
```bash
cp minos/sim_args.py ~/git/minos/minos/config
cp minos/pygame_client.py ~/git/minos/minos/tools
```

## Running the program

```bash
python3 -m minos.tools.pygame_client --depth --rightcamera 'True' --depthright 'True' --save_toc 'True' --save_rootdir

python3 -m minos.tools.pygame_client --depth --rightcamera 'True' --depthright 'True'
```

__NOTE:__ The direction vector is a normalized vector in MINOS world coordinate frame which indicates the direction the agent is facing. The agent uses a coordinate frame with Y up and -Z front. MINOS uses a world coordinate frame with the same conventions: Y up and -Z front. Both SUNCG and Matterport3d scenes are rotated to match this (SUNCG from Y up, +Z front and Matterport3D Z up, -X front, the front is somewhat arbitrary). The Y component of the orientation should be 0 since the agent is just moving in the XZ plane. If you want the relative orientation of the agent to the goal, you will find that in `observation.measurements.direction_to_goal`.
