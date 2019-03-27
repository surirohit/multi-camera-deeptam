copy sim_args.py and sensors.yml to minos/config
copy pygame_client.py to minos/tools
Run the program by
python3 -m minos.tools.pygame_client --depth --rightcamera 'True' --depthright 'True' --save_toc 'True' --save_rootdir

python3 -m minos.tools.pygame_client --depth --rightcamera 'True' --depthright 'True'

The direction vector is a normalized vector in Minos world coordinate frame indicating the direction the agent is facing. The agent uses a coordinate frame with Y up and -Z front. Minos uses a world coordinate frame with the same conventions: Y up and -Z front. Both SUNCG and Matterport3d scenes are rotated to match this (SUNCG from Y up, +Z front and Matterport3D Z up, -X front, the front is somewhat arbitrary). The Y component of the orientation should be 0 since the agent is just moving in the XZ plane. If you want the relative orientation of the agent to the goal, you will find that in observation.measurements.direction_to_goal
