import yaml
import os
from minieigen import Quaternion
from deeptam_tracker.utils.datatypes import Pose
import numpy as np

from deeptam_tracker.utils import message as mg

PRINT_PREFIX = '[UTILS][PARSER]: '


def load_multi_cam_config_yaml(filename):
    """
    Reads a YAML file safely
    :param filename:
    :return:
    """
    data = None
    try:
        with open(filename, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(PRINT_PREFIX, e)
                exit()
    except FileNotFoundError:
        mg.print_fail(PRINT_PREFIX, "Config file not found!")

    ## get path to directory containing the YAML file
    # if absolute path not provided then use relative path with respect to the configuration file location
    for i, config_filename in enumerate(data.get('camera_configs')):
        if not os.path.isabs(config_filename):
            data['camera_configs'][i] = os.path.join(os.path.dirname(os.path.realpath(filename)), config_filename)
            assert os.path.isfile(data['camera_configs'][i])

    return data

def write_tum_trajectory_file(file_path, stamps, poses, head='# timestamp x y z qx qy qz qw\n'):
    """
    :param file_path: desired text file for trajectory (string or handle)
    :param stamps: list of timestamps
    :param poses: list of named tuple poses
    :param head: str text to print on the file
    """
    assert isinstance(stamps, list)
    assert isinstance(poses, list)
    assert len(poses) == len(stamps)

    if len(stamps) < 1 or len(poses) < 1:
        raise Exception(PRINT_PREFIX + 'Input trajectory data has invalid length.')

    assert isinstance(poses[-1], Pose)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(head)
        for i in range(len(poses)):
            pose = poses[i]
            timestamp = stamps[i]

            T = np.eye(4)
            T[:3, :3] = np.array(pose.R)
            T[:3, 3] = np.array(pose.t)
            T = np.linalg.inv(T)  # convert to cam to world
            R = T[:3, :3]
            t = T[:3, 3]

            q = Quaternion(R)
            file.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(timestamp, *t, *q))

    mg.print_notify("Trajectory saved to: " + file_path)

# EOF
