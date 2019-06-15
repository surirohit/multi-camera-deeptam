import matplotlib.pyplot as plt
from PIL import ImageChops
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from deeptam_tracker.utils.vis_utils import convert_between_c2w_w2c, convert_array_to_colorimg

PRED_COLOR_LIST = ['r--', 'g--', 'c--', 'm--', 'y--']
GT_COLOR = 'k'


class Visualizer:
    """
    A class to make visualization of multi-cam DeepTAM tracker simpler and easier.
    """

    def __init__(self, title="DeepTAM Tracker", number_of_cameras=1):
        self.title = title
        self.number_of_cameras = number_of_cameras

        # create figure
        self.fig = plt.figure()
        self.fig.set_size_inches(10.5, 8.5)
        self.fig.suptitle(self.title, fontsize=16)
        # empty list of axes
        self.axs = []

    def startup(self):

        # create axes for plotting the odometry
        ax1 = self.fig.add_subplot(2, 1, 1, projection='3d', aspect='equal')
        # for estimated trajectory
        for idx in range(self.number_of_cameras):
            ax1.plot([], [], [],
                     PRED_COLOR_LIST[idx],
                     label='Prediction (Cam %d)' % idx)
        # for groundtruth
        ax1.plot([], [], [],
                 GT_COLOR,
                 label='Ground truth')
        ax1.legend()
        ax1.set_title('Trajectory')
        self.axs.append(ax1)

        # create axes for RGB image from each camera
        for idx in range(self.number_of_cameras):
            ax = self.fig.add_subplot(2, self.number_of_cameras, self.number_of_cameras + idx + 1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title('RGB (Cam %d)' % idx)
            self.axs.append(ax)

    def update(self, pr_poses_list, gt_poses, frame_list):
        # update groundtruth
        # convert poses from cam2world to world2cam frame
        gt_poses_c2w = [convert_between_c2w_w2c(x) for x in gt_poses]
        self.axs[0].plot(np.array([x.t[0] for x in gt_poses_c2w]),
                         np.array([x.t[1] for x in gt_poses_c2w]),
                         np.array([x.t[2] for x in gt_poses_c2w]),
                         GT_COLOR,
                         label='Ground truth')

        # update trajectory plot
        for idx in range(self.number_of_cameras):
            pr_poses = pr_poses_list[idx]
            pr_poses_c2w = [convert_between_c2w_w2c(x) for x in pr_poses]
            self.axs[0].plot(np.array([x.t[0] for x in pr_poses_c2w]),
                             np.array([x.t[1] for x in pr_poses_c2w]),
                             np.array([x.t[2] for x in pr_poses_c2w]),
                             PRED_COLOR_LIST[idx],
                             label='Prediction (Cam %d)' % idx)

        # update rgb images
        for idx in range(self.number_of_cameras):
            ax_im = self.axs[1 + idx]
            ax_im.cla()
            ax_im.set_title('RGB (Cam %d)' % idx)
            # plot image
            image_cur = convert_array_to_colorimg(frame_list[idx]['image'].squeeze())
            ax_im.imshow(np.array(image_cur))

        plt.pause(1e-9)

# EOF
