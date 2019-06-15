import yaml
import os

from deeptam_tracker.utils import message as mg

PRINT_PREFIX = '[UTILS][PARSER]: '

def pretty_print_nested_dict(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print_nested_dict(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))

def load_camera_config_yaml(config_file):
    """Returns the dictionary created out of parsed YAML file for a single camera

    config_file: str
        Path to the configuration YAML file
    """
    if os.path.exists(config_file):
        file = open(config_file, 'r')
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            mg.print_warn(exc)
            raise Exception(PRINT_PREFIX + "[ERROR] The file is not a valid YAML file!")
        # print success status
        mg.print_pass(PRINT_PREFIX, "Successfully read the camera parameters: %s" % config['cam_name'])

    else:
        raise Exception(PRINT_PREFIX + "[ERROR] YAML file not detected: {0}".format(config_file))

    ## get path to directory containing the YAML file
    # if absolute path not provided then use relative path with respect to the configuration file location
    if config.get('cam_dir', None) is None or not os.path.isabs(config['cam_dir']):
        config['cam_dir'] = os.path.join(os.path.dirname(os.path.realpath(config_file)), config['cam_dir'])

    # throw error if directory does not exist
    if not os.path.isdir(config['cam_dir']):
        raise Exception(PRINT_PREFIX + "[ERROR] Could not find the data directory: %s!" % config['cam_dir'])

    # create a dictionary of dictionaries
    # Reason: YAML parser creates a list out of every element in the parent tag
    for key in config:
        if isinstance(config[key], list):
            child_dict = {}
            for item in config[key]:
                sub_child_dict = {}
                for sub_item in item:
                    if isinstance(item[sub_item] , list):
                        for sub_sub_item in item[sub_item]:
                            sub_child_dict.update(sub_sub_item)
                        child_dict[sub_item] = sub_child_dict
                    else:
                        child_dict.update(item)
                        break
            config[key] = child_dict

    return config
