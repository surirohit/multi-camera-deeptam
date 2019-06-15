import os
import argparse
import matplotlib.pyplot as plt

import deeptam_tracker.models.networks
from deeptam_tracker.evaluation.metrics import rgbd_rpe
from deeptam_tracker.utils import message as mg

from multicam_tracker.utils.parser import load_multi_cam_config_yaml, write_tum_trajectory_file
from multicam_tracker.multicam_tracker import MultiCamTracker
from multicam_tracker.utils.visualizer import Visualizer

PRINT_PREFIX = '[MAIN]: '


def parse_args():
    """
    Parses CLI arguments applicable for this helper script
    """
    # Create parser instance
    parser = argparse.ArgumentParser(description="Run DeepTAM on a multi-camera sequence.")
    # Define arguments
    parser.add_argument('--config_file', '-f', metavar='',
                        help='set to the path to configuration YAML file')
    parser.add_argument('--output_dir', '-o', metavar='',
                        help='set to the path to output directory', default='eval')
    parser.add_argument('--weights', '-w', metavar='',
                        help='set to the path for the weights of the DeepTAM tracking network (without the .index, .meta or .data extensions)')
    parser.add_argument('--tracking_network', '-n', metavar='',
                        help='set to the path of the tracking network (default: path to module deeptam_tracker.models.networks)',
                        default=None)
    parser.add_argument('--disable_vis', '-v', help='disable the frame-by-frame visualization for speed-up',
                        action='store_true')
    # Retrieve arguments
    args = parser.parse_args()
    return args


def track_multicam_rgbd_sequence(checkpoint, config, tracking_module_path, visualization, output_dir):
    """Tracks a multicam rgbd sequence using deeptam tracker
    
    checkpoint: str
        directory to the weights
    
    config: dict
        dictionary containing the list of camera config files

    tracking_module_path: str
        file which contains the model class
        
    visualization: bool


    output_dir: str
        directory path save the output data
    """

    ## initialization
    # initialize the multi-camera sequence

    multicam_tracker = MultiCamTracker(config['camera_configs'], tracking_module_path, checkpoint,
                                       seq_name=config['seq_name'])
    multicam_tracker.startup()

    viz = Visualizer("MultiCam Tracker", len(config['camera_configs']))
    if visualization:
        viz.startup()

    # Putting in higher scope so that don't need to call function again after loop
    pr_poses_list = None
    frame_list = None

    # Reference camera configuration
    try:
        camera_ref_idx = config['camera_ref_index']
    except KeyError:
        camera_ref_idx = 0

    for frame_idx in range(multicam_tracker.get_sequence_length()):

        print(PRINT_PREFIX, 'Input frame number: {}'.format(frame_idx))
        pr_poses_list, gt_poses_list, frame_list, result_list = \
            multicam_tracker.update(frame_idx)

        if visualization:
            viz.update(pr_poses_list, gt_poses_list[camera_ref_idx], frame_list)

    gt_poses_list = multicam_tracker.get_gt_poses_list()
    timestamps_list = multicam_tracker.get_timestamps_list()

    for idx in range(multicam_tracker.num_of_cams):
        ## evaluation
        errors_rpe = rgbd_rpe(gt_poses_list[idx], pr_poses_list[idx], timestamps_list[idx])
        print(PRINT_PREFIX, "Camera %d:" % idx)
        mg.print_notify('Frame-to-keyframe odometry evaluation [RPE], translational RMSE: {}[m/s]'.format(
            errors_rpe['translational_error.rmse']))

        ## save trajectory files
        name = multicam_tracker.cameras_list[idx].name
        write_tum_trajectory_file(os.path.join(output_dir, name, 'stamped_traj_estimate.txt'), timestamps_list[idx],
                                  pr_poses_list[idx])
        write_tum_trajectory_file(os.path.join(output_dir, name, 'stamped_groundtruth.txt'), timestamps_list[idx],
                                  gt_poses_list[idx])

    # final visualization
    viz.update(pr_poses_list, gt_poses_list[camera_ref_idx], frame_list)
    plt.show()

    multicam_tracker.delete_tracker()


def main(args):
    visualization = not args.disable_vis
    config_file = args.config_file
    tracking_module_path = args.tracking_network
    checkpoint = os.path.realpath(args.weights)
    output_dir = os.path.abspath(args.output_dir)

    # read the tracking network path :O
    if tracking_module_path is None:
        tracking_module_path = os.path.abspath(deeptam_tracker.models.networks.__file__)
        mg.print_notify(PRINT_PREFIX, "Using default argument for tracking_network: %s" % tracking_module_path)
    elif not os.path.isfile(tracking_module_path):
        raise Exception(PRINT_PREFIX, "Could not find the network for tracking module: %s!" % tracking_module_path)
    else:
        tracking_module_path = os.path.realpath(tracking_module_path)

    # read the config YAML file and create a dictionary out of it
    config = load_multi_cam_config_yaml(config_file)
    os.makedirs(output_dir, exist_ok=True)

    track_multicam_rgbd_sequence(checkpoint=checkpoint, config=config, tracking_module_path=tracking_module_path,
                                 visualization=visualization, output_dir=output_dir)


if __name__ == "__main__":
    # Retrieve arguments
    ARGS = parse_args()
    # Run main function
    main(ARGS)
