import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageChops
from mpl_toolkits.mplot3d import Axes3D
from minieigen import Vector3, Matrix3, Quaternion
from deeptam_tracker.utils.datatypes import Pose

from deeptam_tracker.tracker import Tracker
import deeptam_tracker.models.networks
from deeptam_tracker.evaluation.rgbd_sequence import RGBDSequence
from deeptam_tracker.evaluation.metrics import rgbd_rpe
from deeptam_tracker.utils.vis_utils import convert_between_c2w_w2c, convert_array_to_colorimg
from deeptam_tracker.utils.parser import load_camera_config_yaml
from deeptam_tracker.utils import message as mg

from multicam_tracker.utils.parser import write_tum_trajectory_file

PRINT_PREFIX = '[MAIN]: '


def parse_args():
    """
    Parses CLI arguments applicable for this helper script
    """
    # Create parser instance
    parser = argparse.ArgumentParser(description="Run DeepTAM on a monocular camera sequence.")
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
    parser.add_argument('--enable_gt', '-e', help='Enable the ground truth', action='store_true')

    # Retrieve arguments
    args = parser.parse_args()
    return args


def init_visualization(enable_gt, title='DeepTAM Tracker'):
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

    if enable_gt:
        ax1.plot([], [], [],
                'g',
                label='Ground truth')

    ax1.legend()
    # ax1.set_zlim(0.5, 1.8)
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


def update_visualization(axes, pr_poses, gt_poses, image_cur, image_cur_virtual, enable_gt):
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

    if enable_gt:
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


def naive_pose_fusion(cams_poses):
    '''
    Averages the input poses of each camera provided in the list

    :param cams_poses: list of list of poses for each camera
    :return: list of poses after fusion
    '''
    from deeptam_tracker.utils.rotation_conversion import rotation_matrix_to_angleaxis, angleaxis_to_rotation_matrix
    from deeptam_tracker.utils.datatypes import Vector3, Matrix3, Pose

    assert isinstance(cams_poses, list)
    assert all(len(cam_poses) == len(cams_poses[0]) for cam_poses in cams_poses)

    fused_poses = []
    num_of_poses = len(cams_poses[0])

    for idx in range(num_of_poses):
        trans = []
        orientation_aa = []
        for cam_num in range(len(cams_poses)):
            trans.append(np.array(cams_poses[cam_num][idx].t))
            orientation_aa.append(rotation_matrix_to_angleaxis(cams_poses[cam_num][idx].R))

        t = np.mean(trans, axis=0)
        R = angleaxis_to_rotation_matrix(Vector3(np.mean(orientation_aa, axis=0)))
        fused_poses.append(Pose(R=Matrix3(R), t=Vector3(t)))

    return fused_poses

def transform_poses_to_base(config, poses):
    """
    Transform the poses to the base frame with settings specified in the config
    :param config:
    :param poses:
    :return:
    """
    ## initialize base to cam transformation
    tf_t_BC = Vector3(config['base_to_cam_pose']['translation']['x'], config['base_to_cam_pose']['translation']['y'],
                      config['base_to_cam_pose']['translation']['z'])
    tf_q_BC = Quaternion(config['base_to_cam_pose']['orientation']['w'], config['base_to_cam_pose']['orientation']['x'],
                         config['base_to_cam_pose']['orientation']['y'], config['base_to_cam_pose']['orientation']['z'])
    tf_R_BC = tf_q_BC.toRotationMatrix()

    ## transformation of poses
    tf_poses = []
    for pose in poses:
        t = pose.t - tf_t_BC
        R = pose.R * tf_R_BC.inverse()
        tf_pose = Pose(R=R, t=t)
        tf_poses.append(tf_pose)

    return tf_poses

def track_rgbd_sequence(checkpoint, config, tracking_module_path, visualization, output_dir, enable_gt):
    """Tracks a rgbd sequence using deeptam tracker
    
    checkpoint: str
        directory to the weights
    
    config: dict
        dictionary containing all the parameters for rgbd sequence and tracker

    tracking_module_path: str
        file which contains the model class
        
    visualization: bool

    output_dir: str
        directory path save the output data
    """

    ### initialization
    # initialize the camera sequence
    sequence = RGBDSequence(config['cam_dir'], rgb_parameters=config['rgb_parameters'],
                            depth_parameters=config['depth_parameters'],
                            time_syncing_parameters=config['time_syncing_parameters'])
    intrinsics = sequence.get_original_normalized_intrinsics()

    # initialize corresponding tracker
    tracker = Tracker(tracking_module_path, checkpoint, intrinsics, tracking_parameters=config['tracking_parameters'])

    gt_poses = []
    timestamps = []
    key_pr_poses = []
    key_gt_poses = []
    key_timestamps = []

    axes = init_visualization(enable_gt)

    frame = sequence.get_dict(0, intrinsics, tracker.image_width, tracker.image_height)
    pose0_gt = frame['pose']
    tracker.clear()
    # WIP: If gt_poses is aligned such that it starts from identity pose, you may comment this line
    # TODO: @Rohit, should we make this base-to-cam transformation?
    tracker.set_init_pose(pose0_gt)

    ## track a sequence
    result = {}
    for frame_idx in range(sequence.get_sequence_length()):
        print(PRINT_PREFIX, 'Input frame number: {}'.format(frame_idx))
        frame = sequence.get_dict(frame_idx, intrinsics, tracker.image_width, tracker.image_height)
        timestamps.append(sequence.get_timestamp(frame_idx))
        result = tracker.feed_frame(frame['image'], frame['depth'])
        gt_poses.append(frame['pose'])
        pr_poses = tracker.poses

        # transform pose to base frame
        print("Before: ", pr_poses[-1].t)
        pr_poses = transform_poses_to_base(config, pr_poses)
        print("After: ", pr_poses[-1].t)
        print('----------------')
        if visualization:
            update_visualization(axes, pr_poses, gt_poses, frame['image'], result['warped_image'], enable_gt)

        if result['keyframe']:
            key_pr_poses.append(tracker.poses[-1])
            key_gt_poses.append(frame['pose'])
            key_timestamps.append(sequence.get_timestamp(frame_idx))

    # transform pose to base frame
    pr_poses = tracker.poses
    pr_poses = transform_poses_to_base(config, pr_poses)

    ## evaluation
    errors_rpe = rgbd_rpe(gt_poses, pr_poses, timestamps)
    mg.print_notify(PRINT_PREFIX, 'Frame-to-keyframe odometry evaluation [RPE], translational RMSE: {}[m/s]'.format(
        errors_rpe['translational_error.rmse']))

    ## fuse the poses naively
    fused_poses = naive_pose_fusion([gt_poses, pr_poses])
    errors_rpe = rgbd_rpe(gt_poses, fused_poses, timestamps)
    mg.print_notify(PRINT_PREFIX,
                    'After fusion, frame-to-keyframe odometry evaluation [RPE], translational RMSE: {}[m/s]'.format(
                        errors_rpe['translational_error.rmse']))

    ## save trajectory files
    write_tum_trajectory_file(os.path.join(output_dir, sequence.cam_name, 'stamped_traj_estimate.txt'), timestamps, pr_poses)
    write_tum_trajectory_file(os.path.join(output_dir, sequence.cam_name, 'stamped_groundtruth.txt'), timestamps, gt_poses)

    ## update visualization
    update_visualization(axes, pr_poses, gt_poses, frame['image'], result['warped_image'], enable_gt)
    plt.show()

    del tracker


def main(args):
    visualization = not args.disable_vis
    config_file = args.config_file
    tracking_module_path = args.tracking_network
    checkpoint = os.path.realpath(args.weights)
    output_dir = os.path.abspath(args.output_dir)
    enable_gt = args.enable_gt

    # read the tracking network path :O
    if tracking_module_path is None:
        tracking_module_path = os.path.abspath(deeptam_tracker.models.networks.__file__)
        mg.print_notify(PRINT_PREFIX, "Using default argument for tracking_network: %s" % tracking_module_path)
    elif not os.path.isfile(tracking_module_path):
        raise Exception(PRINT_PREFIX, "Could not find the network for tracking module: %s!" % tracking_module_path)
    else:
        tracking_module_path = os.path.realpath(tracking_module_path)

    # read the config YAML file and create a dictionary out of it
    config = load_camera_config_yaml(config_file)
    os.makedirs(output_dir, exist_ok=True)

    track_rgbd_sequence(checkpoint=checkpoint, config=config, tracking_module_path=tracking_module_path,
                        visualization=visualization, output_dir=output_dir, enable_gt=enable_gt)


if __name__ == "__main__":
    # Retrieve arguments
    ARGS = parse_args()
    # Run main function
    main(ARGS)
