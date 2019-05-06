import os
import numpy as np
from PIL import Image
import glob
import os
import collections
from deeptam_tracker.evaluation.rgbd_sequence import RGBDSequence
from deeptam_tracker.utils.parser import load_yaml_file
from multicam_tracker.single_cam_tracker import SingleCamTracker


class MultiCamTracker:

    # TODO: require_depth, require_pose
    def __init__(self, config_dirs_list, tracking_module_path, checkpoint):
        """
        Create an object for accessing a multi-camera RGBD sequences

        :param sequence_dirs: list of paths to yaml files containing 
                              location of data directories for each 
                              camera
        :param require_depth:
        :param require_pose:
        """

        # anything with _list at the end of the names has information
        # about cam_i at idx i

        assert isinstance(config_dirs_list, list)

        self.config_dirs_list = config_dirs_list

        self.num_of_cams = len(config_dirs_list)
        print("Setting up trackers for %d cameras." % self.num_of_cams)

        self.cameras_list = []

        # iterate over each directory and write file path names
        for idx in range(self.num_of_cams):
            config = load_yaml_file(config_dirs_list[idx])
            self.cameras_list.append(SingleCamTracker(config, tracking_module_path, checkpoint))

        self.gt_poses = [[] for idx in range(self.num_of_cams)]
        self.timestamps_list = [[] for idx in range(self.num_of_cams)]
        self.key_pr_poses_list = [[] for idx in range(self.num_of_cams)]
        self.key_gt_poses_list = [[] for idx in range(self.num_of_cams)]
        self.key_timestamps_list = [[] for idx in range(self.num_of_cams)]

    def startup(self):
        for cam in self.cameras_list:
            cam.startup()

    def get_sequence_length(self):
        return self.cameras_list[0].get_sequence_length()

    def update(self, frame_idx):

        pr_poses_list = [None for idx in range(self.num_of_cams)]
        frame_list = [None for idx in range(self.num_of_cams)]
        result_list = [None for idx in range(self.num_of_cams)]

        for idx, cam in enumerate(self.cameras_list):
            frame, result, timestamp, pr_poses = cam.update(frame_idx)

            pr_poses_list[idx] = pr_poses
            frame_list[idx] = frame
            result_list[idx] = result

            self.timestamps_list[idx].append(timestamp)
            self.gt_poses[idx].append(frame['pose'])

            if result['keyframe']:
                self.key_pr_poses_list[idx].append(pr_poses[-1])
                self.key_gt_poses_list[idx].append(frame['pose'])
                self.key_timestamps_list[idx].append(timestamp)

        return pr_poses_list, self.gt_poses, frame_list, result_list

    def get_gt_poses_list(self):
        return self.gt_poses

    def get_timestamps_list(self):
        return self.timestamps_list

    def delete_tracker(self):
        for cam in self.cameras_list:
            cam.delete_tracker()
