import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageChops
from mpl_toolkits.mplot3d import Axes3D
import yaml

from deeptam_tracker.tracker import Tracker
import deeptam_tracker.models.networks
from deeptam_tracker.evaluation.rgbd_sequence import RGBDSequence
from deeptam_tracker.evaluation.metrics import rgbd_rpe
from deeptam_tracker.utils.vis_utils import convert_between_c2w_w2c, convert_array_to_colorimg
from deeptam_tracker.utils import message as mg

from multicam_tracker.multicam_tracker import MultiCamTracker

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


def init_visualization(title='DeepTAM Tracker'):
    """Initializes a simple visualization for tracking
    
    title: str
    """
    fig = plt.figure()
    fig.set_size_inches(10.5, 8.5)
    fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(2, 2, 1, projection='3d', aspect='equal')
    ax1.plot([], [], [],
             'r',
             label='Prediction')

    ax1.plot([], [], [],
             'g',
             label='Ground truth')
    ax1.legend()
    ax1.set_zlim(0.5, 1.8)
    ax1.set_title('Trajectory')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax2.set_title('Current image')
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    ax3.set_title('Virtual current image')
    ax4 = fig.add_subplot(2, 2, 3)

    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_title('Diff image')

    return [ax1, ax2, ax3, ax4]


def update_visualization(axes, pr_poses, gt_poses, image_cur, image_cur_virtual):
    """ Updates the visualization for tracking
    
    axes: a list of plt.axes
    
    pr_poses, gt_poses: a list of Pose
    
    image_cur, image_cur_virtual: np.array
    
    """
    pr_poses_c2w = [convert_between_c2w_w2c(x) for x in pr_poses]
    gt_poses_c2w = [convert_between_c2w_w2c(x) for x in gt_poses]

    axes[0].plot(np.array([x.t[0] for x in pr_poses_c2w]),
                 np.array([x.t[1] for x in pr_poses_c2w]),
                 np.array([x.t[2] for x in pr_poses_c2w]),
                 'r',
                 label='Prediction')

    axes[0].plot(np.array([x.t[0] for x in gt_poses_c2w]),
                 np.array([x.t[1] for x in gt_poses_c2w]),
                 np.array([x.t[2] for x in gt_poses_c2w]),
                 'g',
                 label='Ground truth')

    if image_cur_virtual is not None:
        image_cur = convert_array_to_colorimg(image_cur.squeeze())
        image_cur_virtual = convert_array_to_colorimg(image_cur_virtual.squeeze())
        diff = ImageChops.difference(image_cur, image_cur_virtual)
        axes[1].cla()
        axes[1].set_title('Current image')
        axes[2].cla()
        axes[2].set_title('Virtual current image')
        axes[3].cla()
        axes[3].set_title('Diff image')
        axes[1].imshow(np.array(image_cur))
        axes[2].imshow(np.array(image_cur_virtual))
        axes[3].imshow(np.array(diff))

    plt.pause(1e-9)


def update_visualization_all(axes_list, pr_poses_list, gt_poses_list, frame_list, result_list):
    for idx in range(len(axes_list)):
        update_visualization(axes_list[idx],
                             pr_poses_list[idx],
                             gt_poses_list[idx],
                             frame_list[idx]['image'],
                             result_list[idx]['warped_image'])


def track_multicam_rgbd_sequence(checkpoint, config, tracking_module_path, visualization):
    """Tracks a multicam rgbd sequence using deeptam tracker
    
    checkpoint: str
        directory to the weights
    
    config: dict
        dictionary containing the list of camera config files

    tracking_module_path: str
        file which contains the model class
        
    visualization: bool
    """

    ## initialization
    # initialize the multi-camera sequence

    multicam_tracker = MultiCamTracker(config['camera_configs'], tracking_module_path, checkpoint)

    axes_list = [init_visualization(title="DeepTAM Tracker Cam %d" % idx) \
                 for idx in range(len(config['camera_configs']))]

    # Putting in higher scope so that don't need to call function again after loop
    pr_poses_list = None
    gt_poses_list = None
    frame_list = None
    result_list = None

    for frame_idx in range(multicam_tracker.get_sequence_length()):

        print(PRINT_PREFIX, 'Input frame number: {}'.format(frame_idx))
        pr_poses_list, gt_poses_list, frame_list, result_list = \
            multicam_tracker.update(frame_idx)

        # TODO: visualization
        if visualization:
            update_visualization_all(axes_list, pr_poses_list, gt_poses_list, frame_list, result_list)

    gt_poses_list = multicam_tracker.get_gt_poses_list()
    timestamps_list = multicam_tracker.get_timestamps_list()

    for idx in range(len(pr_poses_list)):
        ## evaluation
        errors_rpe = rgbd_rpe(gt_poses_list[idx], pr_poses_list[idx], timestamps_list[idx])
        print(PRINT_PREFIX, "Camera %d:" % idx, )
        mg.print_notify('Frame-to-keyframe odometry evaluation [RPE], translational RMSE: {}[m/s]'.format(
            errors_rpe['translational_error.rmse']))

    # TODO: visualization
    update_visualization_all(axes_list, pr_poses_list, gt_poses_list, frame_list, result_list)
    plt.show()

    multicam_tracker.delete_tracker()


def load_yaml(filename):
    data = None

    try:
        with open(filename, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(PRINT_PREFIX, e)
                exit()
    except FileNotFoundError:
        print(PRINT_PREFIX, "Config file not found")
        exit()
    return data


def main(args):
    visualization = not args.disable_vis
    config_file = args.config_file
    tracking_module_path = args.tracking_network
    checkpoint = os.path.realpath(args.weights)

    # read the tracking network path :O
    if tracking_module_path is None:
        tracking_module_path = os.path.abspath(deeptam_tracker.models.networks.__file__)
        mg.print_notify(PRINT_PREFIX, "Using default argument for tracking_network: %s" % tracking_module_path)
    elif not os.path.isfile(tracking_module_path):
        raise Exception(PRINT_PREFIX, "Could not find the network for tracking module: %s!" % tracking_module_path)
    else:
        tracking_module_path = os.path.realpath(tracking_module_path)

    # read the config YAML file and create a dictionary out of it
    config = load_yaml(config_file)

    track_multicam_rgbd_sequence(checkpoint=checkpoint, config=config, tracking_module_path=tracking_module_path,
                                 visualization=visualization)


if __name__ == "__main__":
    # Retrieve arguments
    ARGS = parse_args()
    # Run main function
    main(ARGS)
