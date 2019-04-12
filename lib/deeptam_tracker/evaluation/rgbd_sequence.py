import os
import numpy as np
import cv2
from PIL import Image
import yaml
from minieigen import Quaternion

from deeptam_tracker.evaluation.rgbd_benchmark.associate import *
from deeptam_tracker.evaluation.rgbd_benchmark.evaluate_rpe import transform44
from deeptam_tracker.utils.datatypes import *
from deeptam_tracker.utils.view_utils import adjust_intrinsics
from deeptam_tracker.utils.rotation_conversion import *
from deeptam_tracker.utils import message as mg


class RGBDSequence:

    def __init__(self, sequence_dir, cam_name='cam', rgb_parameters=None, depth_parameters=None,
                 time_syncing_parameters=None, require_depth=False, require_pose=False):
        """
        Creates an object for accessing an rgbd benchmark sequence

        :param sequence_dir: (str) Path to the directory of a sequence
        :param cam_name: (str) name of the camera
        :param rgb_parameters: (dict) rgb camera parameters, keys = [f_x, f_y, c_x, c_y, width, height]
        :param depth_parameters: (dict) depth camera parameters, keys = [min, max, scaling]
        :param time_syncing_parameters: (dict) parameters for time-syncing, keys = [max_difference, offset]
        :param require_depth:
        :param require_pose:
        """
        # check inputs are correct
        assert isinstance(sequence_dir, str)
        assert isinstance(cam_name, str)
        print(rgb_parameters)
        assert isinstance(rgb_parameters, dict)
        assert isinstance(depth_parameters, dict)
        assert isinstance(time_syncing_parameters, dict)
        if rgb_parameters is None or depth_parameters is None or time_syncing_parameters is None:
            raise Exception("[ERROR] Input parameters are incorrect!")

        self.sequence_dir = os.path.realpath(sequence_dir)
        self.cam_name = cam_name.lower()
        self.intrinsics = None

        # file names with sequence information
        depth_txt = os.path.join(sequence_dir, 'depth.txt')
        rgb_txt = os.path.join(sequence_dir, 'rgb.txt')
        groundtruth_txt = os.path.join(sequence_dir, 'groundtruth.txt')

        # configuration for time-syncing operation
        time_max_difference = time_syncing_parameters['max_difference']
        time_offset = time_syncing_parameters['offset']

        # configuration for depth camera
        self.depth_min = depth_parameters['min']
        self.depth_max = depth_parameters['max']
        self.depth_scaling = depth_parameters['scaling']

        # configuration for rgb camera
        self.intrinsics = [rgb_parameters['f_x'], rgb_parameters['f_y'], rgb_parameters['c_x'], rgb_parameters['c_y']]
        self.original_image_size = (rgb_parameters['width'], rgb_parameters['height'])

        # read paths for rgb and depth images
        self.rgb_dict = read_file_list(rgb_txt)
        self.depth_dict = read_file_list(depth_txt)
        mg.print_notify("Length of the read image sequence: %d" % len(self.rgb_dict))

        # associate two dictionaries of (stamp,data) for rgb and depth data
        self.matches_depth = associate(self.rgb_dict, self.depth_dict, offset=time_offset,
                                       max_difference=time_max_difference)
        self.matches_depth_dict = dict(self.matches_depth)

        # create the camera matrix
        self._K = np.eye(3)
        self._K[0, 0] = self.intrinsics[0]
        self._K[1, 1] = self.intrinsics[1]
        self._K[0, 2] = self.intrinsics[2]
        self._K[1, 2] = self.intrinsics[3]

        # read groundtruth if available
        if os.path.isfile(groundtruth_txt):
            self.groundtruth_dict = read_file_list(groundtruth_txt)
        else:
            self.groundtruth_dict = None
        # associate two dictionaries of (stamp,data) for rgb and groundtruth data
        if self.groundtruth_dict is not None:
            self.matches_pose = associate(self.rgb_dict, self.groundtruth_dict, offset=time_offset,
                                          max_difference=time_max_difference)
            self.matches_pose_dict = dict(self.matches_pose)
        else:
            self.matches_pose = None
            self.matches_pose_dict = None

        # create list of the processed (time-synced) rgb-depth images and poses on the basis of rgb timestamps
        # Note: if any dictionary is empty, None is added for the entry
        self.matches_depth_pose = []
        for timestamp_rgb in sorted(self.rgb_dict.keys()):
            # RGB image
            img_path = os.path.join(self.sequence_dir, *self.rgb_dict[timestamp_rgb])
            if not os.path.exists(img_path):
                continue
            # Corresponding depth image
            if self.matches_depth_dict.get(timestamp_rgb, None) is not None:
                timestamp_depth = self.matches_depth_dict[timestamp_rgb]
                depth_path = os.path.join(self.sequence_dir, *self.depth_dict[timestamp_depth])
                if not os.path.exists(depth_path):
                    timestamp_depth = None
            else:
                timestamp_depth = None
            if require_depth and timestamp_depth is None:
                continue
            # Corresponding pose
            if self.matches_pose_dict is not None and self.matches_pose_dict.get(timestamp_rgb, None) is not None:
                timestamp_pose = self.matches_pose_dict[timestamp_rgb]
            else:
                timestamp_pose = None
            if require_pose and timestamp_pose is None:
                continue
            # append the timestamps of the synced information
            timestamps_sync = {
                'timestamp_rgb': timestamp_rgb,
                'timestamp_depth': timestamp_depth,
                'timestamp_pose': timestamp_pose
            }
            self.matches_depth_pose.append(timestamps_sync)

        # make sure the initial frame has a depth map and a pose
        while self.matches_depth_pose[0]['timestamp_depth'] is None or self.matches_depth_pose[0][
            'timestamp_pose'] is None:
            del self.matches_depth_pose[0]

        # get the sequence length after processing
        self.seq_len = len(self.matches_depth_pose)
        mg.print_notify("Length of the synced image sequence: %d" % self.seq_len)

        # open first matched image to get the original image size
        im_size = Image.open(os.path.join(self.sequence_dir,
                                          *self.rgb_dict[self.matches_depth_pose[0]['timestamp_rgb']])).size
        if self.original_image_size != im_size:
            raise Exception("Expected input images to be of size ({}, {}) but received ({}, {})" \
                            .format(self.original_image_size[0], self.original_image_size[1],
                                    im_size[0], im_size[1]))

    def name(self):
        """
        Return the name of the camera
        """
        return self.cam_name

    def get_sequence_length(self):
        """
        Returns the sequence length
        """
        return self.seq_len

    def get_timestamp(self, frame):
        """
        Returns the timestamp which corresponds to the rgb frame
        """
        return self.matches_depth_pose[frame]['timestamp_rgb']

    def get_original_normalized_intrinsics(self):
        """
        Returns the original intrinsics in normalized form
        """
        return np.array([
            self._K[0, 0] / self.original_image_size[0],
            self._K[1, 1] / self.original_image_size[1],
            self._K[0, 2] / self.original_image_size[0],
            self._K[1, 2] / self.original_image_size[1]
        ], dtype=np.float32)

    def get_view(self, frame, normalized_intrinsics=None, width=128, height=96, depth=True):
        """Returns a view object for the given rgb frame

        frame: int
            The rgb frame number

        normalized_intrinsics: np.array or list
            Normalized intrinsics. Default is sun3d

        width: int
            image width. default is 128

        height: int
            image height. default is 96

        depth: bool
            If true the returned view object contains the depth map
        """

        if normalized_intrinsics is None:
            normalized_intrinsics = self.get_sun3d_intrinsics()

        new_K = np.eye(3)
        new_K[0, 0] = normalized_intrinsics[0] * width
        new_K[1, 1] = normalized_intrinsics[1] * height
        new_K[0, 2] = normalized_intrinsics[2] * width
        new_K[1, 2] = normalized_intrinsics[3] * height

        # get associated synced timestamps for RGB-Depth-Pose measurements
        timestamps_sync = self.matches_depth_pose[frame]
        trgb = timestamps_sync['timestamp_rgb']
        tdepth = timestamps_sync['timestamp_depth']
        tpose = timestamps_sync['timestamp_pose']

        # read RGB imagee
        img_path = os.path.join(self.sequence_dir, *self.rgb_dict[trgb])
        img = Image.open(img_path)
        img.load()
        # Convert image from RGBA to RGB!
        if np.array(img).shape[2] == 4:
            img = Image.fromarray(cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB))

        # read depth image
        if depth and tdepth is not None:
            depth_path = os.path.join(self.sequence_dir, *self.depth_dict[tdepth])
            dpth = self.read_depth_image(depth_path, self.depth_scaling)
            dpth_metric = 'camera_z'
        else:
            dpth = None
            dpth_metric = None

        # read groundtruth
        if tpose is not None:
            pose_tuple = [tpose] + self.groundtruth_dict[tpose]
            T = transform44(pose_tuple)
            T = np.linalg.inv(T)  # convert to (world to cam)

            R = T[:3, :3]
            t = T[:3, 3]
        else:
            R = np.eye(3)
            t = np.array([0, 0, 0], dtype=np.float)

        view = View(R=R, t=t, K=self._K, image=img, depth=dpth, depth_metric=dpth_metric)

        new_view = adjust_intrinsics(view, new_K, width, height)
        if depth and tdepth is not None:
            d = new_view.depth
            d[d <= 0] = np.nan
            new_view = new_view._replace(depth=d)

        view.image.close()
        del view
        return new_view

    def get_image(self, frame, normalized_intrinsics=None, width=128, height=96):
        """Returns the normalized image [-0/5. 0.5] for the specified frame as numpy array

        frame: int
            The rgb frame number
        
        normalized_intrinsics: np.array or list
            Normalized intrinsics. Default is sun3d

        width: int
            image width. default is 128

        height: int
            image height. default is 96

        """
        img = self.get_view(frame, normalized_intrinsics, width, height, depth=False).image
        img_arr = np.array(img).transpose([2, 0, 1]).astype(np.float32) / 255 - 0.5
        return img_arr

    def get_depth(self, frame, normalized_intrinsics=None, width=None, height=None, inverse=False):
        """Returns the depth for the specified frame

        frame: int
            The rgb frame number
        
        normalized_intrinsics: np.array or list
            Normalized intrinsics. Default is sun3d

        width: int
            image width. default is 128

        height: int
            image height. default is 96

        """
        depth = self.get_view(frame, normalized_intrinsics, width, height, depth=True, ).depth
        if inverse and depth is not None:
            depth = 1 / depth
        return depth

    def get_image_depth(self, frame, normalized_intrinsics=None, width=None, height=None, inverse=False):
        """Returns the depth for the specified frame

        frame: int
            The rgb frame number
        
        normalized_intrinsics: np.array or list
            Normalized intrinsics. Default is sun3d

        width: int
            image width. default is 128

        height: int
            image height. default is 96

        """
        view = self.get_view(frame, normalized_intrinsics, width, height, depth=True)
        depth = view.depth
        if inverse and depth is not None:
            depth = 1 / depth
        return (view.image, depth)

    def get_dict(self, frame, normalized_intrinsics=None, width=128, height=96):
        """Returns image, depth and pose as a dict of numpy arrays
        The depth is the inverse depth.
        
        frame: int
            The rgb frame number

        normalized_intrinsics: np.array
            normalized intrinsics of the camera

        width: int
            image width. default is 128

        height: int
            image height. default is 96
        """
        view = self.get_view(frame, normalized_intrinsics=normalized_intrinsics, width=width, height=height, depth=True)

        img_arr = np.array(view.image).transpose([2, 0, 1]).astype(np.float32) / 255 - 0.5
        rotation = Quaternion(view.R).toAngleAxis()
        rotation = rotation[0] * np.array(rotation[1])

        result = {
            'image': img_arr[np.newaxis, :, :, :],
            'depth': None,
            'rotation': rotation[np.newaxis, :],
            'translation': view.t[np.newaxis, :],
            'pose': Pose(R=Matrix3(angleaxis_to_rotation_matrix(rotation)), t=Vector3(view.t))
        }
        if view.depth is not None:
            result['depth'] = (1 / view.depth)[np.newaxis, np.newaxis, :, :]

        return result

    def get_relative_motion(self, frame1, frame2):
        """Returns the relative transformation from frame1 to frame2

        frame1: int
            Frame number 1

        frame2: int
            Frame number 2
        """
        if self.groundtruth_dict is None:
            return None

        # get associated synced timestamps for RGB-Depth-Pose measurements
        # frame 1:
        timestamp_sync = self.matches_depth_pose[frame1]
        tpose = timestamp_sync['timestamp_pose']
        pose_tuple = [tpose] + self.groundtruth_dict[tpose]
        inv_T1 = transform44(pose_tuple)

        # frame 2:
        timestamp_sync = self.matches_depth_pose[frame2]
        tpose = timestamp_sync['timestamp_pose']
        pose_tuple = [tpose] + self.groundtruth_dict[tpose]
        T2 = transform44(pose_tuple)
        T2 = np.linalg.inv(T2)  # convert to world to cam2

        # compute relative motion
        T = T2.dot(inv_T1)
        R12 = T[:3, :3]
        t12 = T[:3, 3]
        rotation = Quaternion(R12).toAngleAxis()
        rotation = rotation[0] * np.array(rotation[1])

        return {
            'rotation': rotation[np.newaxis, :],
            'translation': t12[np.newaxis, :],
        }

    @staticmethod
    def get_sun3d_intrinsics():
        """Returns the normalized intrinsics of sun3d"""
        return np.array([0.89115971, 1.18821287, 0.5, 0.5], dtype=np.float32)

    @staticmethod
    def read_depth_image(path, scaling_factor=5000.0):
        """Reads a png depth image and returns it as 2d numpy array.
        Invalid values will be represented as NAN

        path: str
            Path to the image

        scaling_factor: float
            Scaling the depth images (default: 5000.0)
        """
        depth = Image.open(path).convert('I')
        depth.load()
        if depth.mode != "I":
            raise Exception("Depth image is not in intensity format: {0}".format(path))
        depth_arr = np.array(depth) / scaling_factor
        del depth
        return depth_arr.astype(np.float32)

    @staticmethod
    def write_rgbd_pose_format(path, poses, timestamps):
        """writes a pose txt file compatible with the rgbd eval tools

        path: str
            Path to the output file

        poses: list of Pose
        timestamps: list of float
        """
        assert len(poses) == len(timestamps)
        with open(path, 'w') as f:
            for i in range(len(poses)):
                pose = poses[i]
                timestamp = timestamps[i]

                T = np.eye(4)
                T[:3, :3] = np.array(pose.R)
                T[:3, 3] = np.array(pose.t)
                T = np.linalg.inv(T)  # convert to cam to world
                R = T[:3, :3]
                t = T[:3, 3]

                q = Quaternion(R)
                f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(timestamp, *t, *q))
