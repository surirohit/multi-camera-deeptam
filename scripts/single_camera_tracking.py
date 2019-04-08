import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageChops
from mpl_toolkits.mplot3d import Axes3D

from deeptam_tracker.tracker import Tracker
import deeptam_tracker.models.networks
from deeptam_tracker.evaluation.rgbd_sequence import RGBDSequence
from deeptam_tracker.evaluation.metrics import rgbd_rpe
from deeptam_tracker.utils.vis_utils import convert_between_c2w_w2c, convert_array_to_colorimg
from deeptam_tracker.utils import message as mg


def parse_args():
    """
    Parses CLI arguments applicable for this helper script
    """
    # Create parser instance
    parser = argparse.ArgumentParser(description="Run benchmarking tasks using noesis applet.")
    # Define arguments
    parser.add_argument('--data_dir', '-d', metavar='',
                        help='set a sequence data directory (should be consistent with TUM RGBD-SLAM datasets)')
    parser.add_argument('--weights', '-w', metavar='',
                        help='set to path for the weights of the DeepTAM tracking network (without the .index, .meta or .data extensions)')
    parser.add_argument('--tracking_network', '-n', metavar='',
                        help='set to path of the tracking network (default: path to module deeptam_tracker.models.networks',
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


def track_rgbd_sequence(checkpoint, datadir, tracking_module_path, visualization):
    """Tracks a rgbd sequence using deeptam tracker
    
    checkpoint: str
        directory to the weights
    
    datadir: str
        directory to the sequence data

    tracking_module_path: str
        file which contains the model class
        
    visualization: bool
    """

    ## initialization
    sequence = RGBDSequence(datadir)
    intrinsics = sequence.get_sun3d_intrinsics()
    tracker = Tracker(tracking_module_path, checkpoint, intrinsics)

    gt_poses = []
    timestamps = []
    key_pr_poses = []
    key_gt_poses = []
    key_timestamps = []

    axes = init_visualization()

    frame = sequence.get_dict(0, intrinsics, tracker.image_width, tracker.image_height)
    pose0_gt = frame['pose']
    tracker.clear()
    # WIP: If gt_poses is aligned such that it starts from identity pose, you may comment this line
    tracker.set_init_pose(pose0_gt)

    ## track a sequence
    result = {}
    for frame_idx in range(sequence.get_sequence_length()):
        print('frame {}'.format(frame_idx))
        frame = sequence.get_dict(frame_idx, intrinsics, tracker.image_width, tracker.image_height)
        timestamps.append(sequence.get_timestamp(frame_idx))
        result = tracker.feed_frame(frame['image'], frame['depth'])
        gt_poses.append(frame['pose'])
        pr_poses = tracker.poses

        if visualization:
            update_visualization(axes, pr_poses, gt_poses, frame['image'], result['warped_image'])

        if result['keyframe']:
            key_pr_poses.append(tracker.poses[-1])
            key_gt_poses.append(frame['pose'])
            key_timestamps.append(sequence.get_timestamp(frame_idx))

    ## evaluation
    pr_poses = tracker.poses
    errors_rpe = rgbd_rpe(gt_poses, pr_poses, timestamps)
    mg.print_notify('Frame-to-keyframe odometry evaluation [RPE], translational RMSE: {}[m/s]'.format(
        errors_rpe['translational_error.rmse']))

    update_visualization(axes, pr_poses, gt_poses, frame['image'], result['warped_image'])
    plt.show()

    del tracker


def main(args):
    visualization = not args.disable_vis
    data_dir = os.path.realpath(args.data_dir)
    checkpoint = os.path.realpath(args.weights)

    # read the tracking network path :O
    if args.tracking_network is None:
        tracking_module_path = os.path.abspath(deeptam_tracker.models.networks.__file__)
        mg.print_notify("Using default argument for tracking_network: %s" % tracking_module_path)
    else:
        tracking_module_path = os.path.realpath(args.tracking_network)

    # ensure correct paths ;)
    if not os.path.isdir(data_dir):
        raise Exception("Could not find the data directory: %s!" % data_dir)
    if not os.path.isfile(tracking_module_path):
        raise Exception("Could not find the network for tracking module: %s!" % tracking_module_path)

    track_rgbd_sequence(checkpoint=checkpoint, datadir=data_dir, tracking_module_path=tracking_module_path,
                        visualization=visualization)


if __name__ == "__main__":
    # Retrieve arguments
    ARGS = parse_args()
    # Run main function
    main(ARGS)
