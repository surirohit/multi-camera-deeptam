import cv2
import numpy as np

from deeptam_tracker.utils.rotation_conversion import rotation_matrix_to_angleaxis, angleaxis_to_rotation_matrix
from deeptam_tracker.utils.datatypes import Vector3, Matrix3, Pose
from deeptam_tracker.utils.vis_utils import convert_between_c2w_w2c, convert_array_to_colorimg
import deeptam_tracker.utils.message as mg

PRINT_PREFIX = "[FUSION]: "


def naive_avg_pose_fusion(cams_poses):
    '''
    Averages the input poses of each camera provided in the list

    :param cams_poses: list of poses for each camera
    :return: pose after fusion
    '''
    from deeptam_tracker.utils.rotation_conversion import rotation_matrix_to_angleaxis, angleaxis_to_rotation_matrix
    from deeptam_tracker.utils.datatypes import Vector3, Matrix3, Pose
    from deeptam_tracker.utils.vis_utils import convert_between_c2w_w2c

    trans = []
    orientation_aa = []
    for cam_num in range(len(cams_poses)):
        # transform pose to the world frame
        pose_c2w = convert_between_c2w_w2c(cams_poses[cam_num])
        # append to the list
        trans.append(np.array(pose_c2w.t))
        orientation_aa.append(rotation_matrix_to_angleaxis(pose_c2w.R))

    # naive approach by taking average
    t = np.mean(trans, axis=0)
    R = angleaxis_to_rotation_matrix(Vector3(np.mean(orientation_aa, axis=0)))
    fused_pose_c2w = Pose(R=Matrix3(R), t=Vector3(t))

    return convert_between_c2w_w2c(fused_pose_c2w)


def sift_pose_fusion(cams_poses, images_list):
    '''
    Averages the input poses of each camera provided in the list based on features density

    :param cams_poses: list of poses for each camera
    :param images_list: list of images from each camera
    :return: pose after fusion
    '''
    assert isinstance(images_list, list)
    assert isinstance(cams_poses, list)
    assert (len(cams_poses) == len(images_list))

    ## SIFT feature detection
    feat_num = []
    sift = cv2.xfeatures2d.SIFT_create()
    for image in images_list:
        im = np.array(convert_array_to_colorimg(image.squeeze()))
        kp = sift.detect(im, None)
        feat_num.append(len(kp))

    feat_num = np.asarray(feat_num)
    feat_weights = feat_num / feat_num.sum()

    trans = []
    orientation_aa = []
    for cam_num in range(len(cams_poses)):
        # transform pose to the world frame
        pose_c2w = convert_between_c2w_w2c(cams_poses[cam_num])
        # append to the list
        trans.append(np.array(pose_c2w.t))
        orientation_aa.append(rotation_matrix_to_angleaxis(pose_c2w.R))

    # naive approach by taking average
    t = np.average(trans, axis=0, weights=feat_weights)
    R = angleaxis_to_rotation_matrix(Vector3(np.average(orientation_aa, axis=0, weights=feat_weights)))
    fused_pose_c2w = Pose(R=Matrix3(R), t=Vector3(t))

    return convert_between_c2w_w2c(fused_pose_c2w)


def sift_depth_pose_fusion(cams_poses, images_list, depths_list):
    '''
    Averages the input poses of each camera provided in the list based on features density
    and homogeneity of the depth images

    :param cams_poses: list of poses for each camera
    :param images_list: list of images from each camera
    :param depths_list: list of depth images from each camera
    :return: pose after fusion
    '''
    assert isinstance(images_list, list)
    assert isinstance(depths_list, list)
    assert isinstance(cams_poses, list)
    assert (len(cams_poses) == len(images_list))
    assert (len(cams_poses) == len(depths_list))

    ## SIFT feature detection
    sift_weights = []
    sift = cv2.xfeatures2d.SIFT_create()
    for image in images_list:
        im = np.array(convert_array_to_colorimg(image.squeeze()))
        kp = sift.detect(im, None)
        sift_weights.append(len(kp))

    sift_weights = np.asarray(sift_weights)
    sift_weights = sift_weights / sift_weights.sum()

    depth_weights = []
    for depth in depths_list:
        depth_weights.append(np.nanstd(depth))
    depth_weights = np.asarray(depth_weights)
    depth_weights = depth_weights / depth_weights.sum()

    c1 = 0.5
    c2 = 0.5
    feat_weights = []
    for weight_idx in range(sift_weights.shape[0]):
        feat_weights.append(c1*sift_weights[weight_idx] + c2*depth_weights[weight_idx])
    feat_weights = np.asarray(feat_weights)
    feat_weights = feat_weights / feat_weights.sum()

    trans = []
    orientation_aa = []
    for cam_num in range(len(cams_poses)):
        # transform pose to the world frame
        pose_c2w = convert_between_c2w_w2c(cams_poses[cam_num])
        # append to the list
        trans.append(np.array(pose_c2w.t))
        orientation_aa.append(rotation_matrix_to_angleaxis(pose_c2w.R))

    # naive approach by taking average
    t = np.average(trans, axis=0, weights=feat_weights)
    R = angleaxis_to_rotation_matrix(Vector3(np.average(orientation_aa, axis=0, weights=feat_weights)))
    fused_pose_c2w = Pose(R=Matrix3(R), t=Vector3(t))

    return convert_between_c2w_w2c(fused_pose_c2w)


def rejection_avg_pose_fusion(cams_poses, amt_dev=1.4):
    '''
    Averages the input poses of each camera provided in the list based on pose(translation only) acceptability
    The acceptability is evaluated using the sigma-based rule for outlier rejection in a data.

    :param cams_poses: list of poses for each camera
    :param amt_dev: number of times of the standard deviation to consider for inlier acceptance
    :return: pose after fusion
    '''
    trans = []
    orientation_aa = []
    for cam_num in range(len(cams_poses)):
        # transform pose to the world frame
        pose_c2w = convert_between_c2w_w2c(cams_poses[cam_num])
        # append to the list
        trans.append(np.array(pose_c2w.t))
        orientation_aa.append(rotation_matrix_to_angleaxis(pose_c2w.R))

    # calculate the mean and standard deviation of positions
    t_mean = np.average(trans, axis=0)
    t_sigma = amt_dev * np.std(trans, axis=0)

    # filtering by outlier removal using 1-sigma rule
    trans = []
    orientation_aa = []
    for cam_num in range(len(cams_poses)):
        # transform pose to the world frame
        pose_c2w = convert_between_c2w_w2c(cams_poses[cam_num])
        trans_c2w = np.array(pose_c2w.t)
        # append to the list if satisfies 1-sigma rule
        if all(np.abs(trans_c2w - t_mean) <= t_sigma):
            trans.append(np.array(pose_c2w.t))
            orientation_aa.append(rotation_matrix_to_angleaxis(pose_c2w.R))
        else:
            mg.print_warn(PRINT_PREFIX, "Camera %d ignored during fusion!" % cam_num)

    # approach by taking average of only filtered poses
    t = np.mean(trans, axis=0)
    R = angleaxis_to_rotation_matrix(Vector3(np.mean(orientation_aa, axis=0)))
    fused_pose_c2w = Pose(R=Matrix3(R), t=Vector3(t))

    return convert_between_c2w_w2c(fused_pose_c2w)

# EOF
