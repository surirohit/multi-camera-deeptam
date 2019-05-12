from deeptam_tracker.utils.parser import load_camera_config_yaml
from multicam_tracker.single_cam_tracker import SingleCamTracker
from deeptam_tracker.utils import message as mg


PRINT_PREFIX = '[MULTICAM TRACKER]: '

class MultiCamTracker:

    def __init__(self, config_dirs_list, tracking_module_path, checkpoint, seq_name='multi_cam'):
        """
        Create an object for accessing a multi-camera RGBD sequences

        :param config_dirs_list: (list) paths to yaml files containing location of
                                 data directories for each  camera
        :param tracking_module_path: (str) path to file containing the Tracker network class
        :param checkpoint: (str) path to the pre-trained network weights
        """

        # anything with _list at the end of the names has information
        # about cam_i at idx i

        assert isinstance(config_dirs_list, list)
        assert len(config_dirs_list) > 0

        self._startup = False
        self.config_dirs_list = config_dirs_list

        self.seq_name = seq_name
        self.num_of_cams = len(config_dirs_list)

        mg.print_notify(PRINT_PREFIX, "Setting up trackers for %d cameras." % self.num_of_cams)
        self.cameras_list = []

        # iterate over each directory and write file path names
        for idx in range(self.num_of_cams):
            config = load_camera_config_yaml(config_dirs_list[idx])
            self.cameras_list.append(SingleCamTracker(config, tracking_module_path, checkpoint))

        self.gt_poses = [[] for _ in range(self.num_of_cams)]
        self.timestamps_list = [[] for _ in range(self.num_of_cams)]
        self.key_pr_poses_list = [[] for _ in range(self.num_of_cams)]
        self.key_gt_poses_list = [[] for _ in range(self.num_of_cams)]
        self.key_timestamps_list = [[] for _ in range(self.num_of_cams)]

    def __del__(self):
        self.delete_tracker()

    def startup(self):
        """
        Sequentially calls the startup() function for all the cameras in the setup
        :return:
        """
        self._startup = True
        for cam in self.cameras_list:
            cam.startup()
            self._startup = self._startup & cam._startup

    def delete_tracker(self):
        for cam in self.cameras_list:
            cam.delete_tracker()

    def update(self, frame_idx):
        """
        Sequentially calls the update() function for all the cameras in the setup
        :param frame_idx: Frame number in the data to perform tracking on
        :return:
            (list) pr_poses_list: list of the predicted poses for all the camera that at frame index
            (list(list)) gt_poses: list of lists of groundtruth poses for all the cameras till that frame index
            (list(dict)) frame_list: list of dict containing the RGB image, depth image and groundtruth pose for
                                     all the cameras
            (list(dict)) result_list: list of dict containing the pose, warped image and keyframe information for
                                      all the cameras
        """
        if not self._startup:
            raise Exception(PRINT_PREFIX + 'Trackers have not been initialized. Please call startup() first.')

        pr_poses_list = [None for _ in range(self.num_of_cams)]
        frame_list = [None for _ in range(self.num_of_cams)]
        result_list = [None for _ in range(self.num_of_cams)]

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

    def get_sequence_length(self):
        return self.cameras_list[0].get_sequence_length()

    def get_gt_poses_list(self):
        return self.gt_poses

    def get_timestamps_list(self):
        return self.timestamps_list

# EOF
