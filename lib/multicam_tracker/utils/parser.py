import yaml
import os

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
