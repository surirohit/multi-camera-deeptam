import os
import numpy as np
from PIL import Image
import glob
import os
import collections
from deeptam_tracker.evaluation.rgbd_sequence import RGBDSequence


class MultiCamSequence:

    def __init__(self, sequence_dirs, require_depth=False, require_pose=False):
        """
        Create an object for accessing a multi-camera RGBD sequences

        :param sequence_dirs: list of paths to directories containing the data for each camera
        :param require_depth:
        :param require_pose:
        """
        assert isinstance(sequence_dirs, list)

        self.sequence_dirs = sequence_dirs
        self.num_of_cams = len(sequence_dirs)

        self.cameras = collections.defaultdict(dict)

        # iterate over each directory and write file path names
        for i in range(self.num_of_cams):
            assert os.path.isdir(self.sequence_dirs[i])
            self.cameras['cam_%d' % i] = RGBDSequence(self.sequence_dirs[i], require_depth, require_pose)
