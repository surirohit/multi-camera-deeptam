from deeptam_tracker.evaluation.rgbd_sequence import RGBDSequence
from deeptam_tracker.tracker import Tracker
from minieigen import Vector3, Quaternion
from deeptam_tracker.utils.datatypes import *

PRINT_PREFIX = '[SINGLECAM TRACKER]: '


class SingleCamTracker:

    # TODO: require_depth, require_pose
    def __init__(self, config, tracking_module_path, checkpoint, require_depth=False, require_pose=False):
        """
        Creates an object for a single camera tracking instance. It wraps around the DeepTAM Sequence and Tracker
        classes. This is done to make it easier to extend the setup for a multi-camera tracking class.

        :param config: (dict) comprising of the camera settings
        :param tracking_module_path: (str) path to file containing the Tracker network class
        :param checkpoint: (str) path to the pre-trained network weights
        :param require_depth:
        :param require_pose:
        """
        assert isinstance(config, dict)

        self._startup = False
        self.name = config['cam_name']

        # base to camera transformation
        self.tf_t_BC = Vector3(config['base_to_cam_pose']['translation']['x'],
                          config['base_to_cam_pose']['translation']['y'],
                          config['base_to_cam_pose']['translation']['z'])
        self.tf_q_BC = Quaternion(config['base_to_cam_pose']['orientation']['w'],
                             config['base_to_cam_pose']['orientation']['x'],
                             config['base_to_cam_pose']['orientation']['y'],
                             config['base_to_cam_pose']['orientation']['z'])
        self.tf_R_BC = self.tf_q_BC.toRotationMatrix()

        self.sequence = RGBDSequence(config['cam_dir'], rgb_parameters=config['rgb_parameters'],
                                     depth_parameters=config['depth_parameters'],
                                     time_syncing_parameters=config['time_syncing_parameters'],
                                     cam_name=config['cam_name'], require_depth=require_depth,
                                     require_pose=require_pose)

        self.intrinsics = self.sequence.get_original_normalized_intrinsics()

        self.tracker = Tracker(tracking_module_path,
                               checkpoint,
                               self.intrinsics,
                               tracking_parameters=config['tracking_parameters'])

    def __del__(self):
        self.delete_tracker()

    def startup(self):
        """
        Clears the tracker and sets the initial pose of the camera
        :return:
        """
        frame = self.sequence.get_dict(0,
                                       self.intrinsics,
                                       self.tracker.image_width,
                                       self.tracker.image_height)

        pose0_gt = frame['pose']

        self.tracker.clear()
        # WIP: If gt_poses is aligned such that it starts from identity pose, you may comment this line
        # TODO: @Rohit, should we make this base-to-cam transformation?
        self.tracker.set_init_pose(pose0_gt)
        self._startup = True

    def delete_tracker(self):
        del self.tracker

    def update(self, frame_idx):
        """
        Performs the tracking for an input image marked by the frame_idx
        :param frame_idx: Frame number in the data to perform tracking on
        :return:
            (dict) frame: contains the RGB image, depth image, and groundtruth pose for that frame index
            (dict) result: contains the pose, warped image and keyframe information
            (float) timestamp: timestamp corresponding to that frame index
            (list) poses: list of all the poses computed so far
        """
        if not self._startup:
            raise Exception(PRINT_PREFIX +
                            'Tracker \"%s\"has not been initialized. Please call startup() first.' % self.name)

        frame = self.sequence.get_dict(frame_idx,
                                       self.intrinsics,
                                       self.tracker.image_width,
                                       self.tracker.image_height)

        result = self.tracker.feed_frame(frame['image'], frame['depth'])

        return frame, result, self.sequence.get_timestamp(frame_idx), self.tracker.poses

    def get_sequence_length(self):
        return self.sequence.get_sequence_length()

    def get_transformed_pose(self):
        """
        Transforms the poses estimated from the tracker to a base frame
        :return:
        """
        ## transformation of poses
        tf_poses = []
        for pose in self.tracker.poses:
            t = self.tf_R_BC * pose.t + self.tf_t_BC
            R = self.tf_R_BC * pose.R
            tf_pose = Pose(R=R, t=t)
            tf_poses.append(tf_pose)

        return tf_poses
# EOF
