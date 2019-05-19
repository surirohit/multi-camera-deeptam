#!/usr/bin/python

import os
import sys
import argparse
import cv2
import re
import yaml

# ROS
import rosbag
import rospy
from cv_bridge import CvBridge, CvBridgeError

##############################################
####### CONFIGURATION FOR TTHE SCRIPT ########
##############################################
CAM_NUM_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
RECORD_CAM_RGB = True
RECORD_CAM_INFO = False
RECORD_CAM_DEPTH = False


##############################################
### DO NOT TOUCH ANYTHING BEYOND THIS PART ###
##############################################

class CameraIndexBag:
    """
    A class to make the code readable?
    """

    def __init__(self, camera_idx, output_path):
        """
        Default constructor
        :param camera_idx: camera index in the bag file
        :param output_path: base directory
        """
        self.idx = camera_idx
        self.output_path = os.path.join(os.path.abspath(output_path), "cam_%d" % self.idx)

        # topic names associated to that camera in the bag file
        self.topics = []
        if RECORD_CAM_RGB:
            self.topics.append("/pair%d/left/image_rect_color" % self.idx)
        if RECORD_CAM_INFO:
            self.topics.append("/uvc_camera/cam_%d/camera_info" % self.idx)
        # if camera is on the left side (i.e. even numbered then it provides the disparity image as well
        if self.idx % 2 == 0 and RECORD_CAM_DEPTH:
            self.topics.append("/uvc_camera/cam_%d/image_depth" % (self.idx))

    def __del__(self):
        """
        Default destructor
        :return:
        """
        if RECORD_CAM_RGB:
            self.rgb_log_file.close()
        if self.idx % 2 == 0 and RECORD_CAM_DEPTH:
            self.depth_log_file.close()

    def create_log_directories(self, rosbag_path):
        """
        Create directories for the camera to dump data in and also create the text files to log the data
        :param rosbag_path: path to the rosbag file
        :return:
        """
        # get the absolute output path
        rosbag_name = os.path.basename(rosbag_path)

        if RECORD_CAM_RGB:
            # path to save the rgb images
            self.camera_path_rgb = os.path.join(self.output_path, 'rgb')
            if not os.path.exists(self.camera_path_rgb):
                os.makedirs(self.camera_path_rgb)

            self.rgb_log_file = open(os.path.join(self.output_path, "rgb.txt"), "w")
            self.rgb_log_file.write("# color images \n" + \
                                    "# file: '%s' \n" % rosbag_name + \
                                    "# timestamp filename \n")

        # path to save the disparity images (if applicable)
        if self.idx % 2 == 0 and RECORD_CAM_DEPTH:
            self.camera_path_depth = os.path.join(self.output_path, 'depth')
            if not os.path.exists(self.camera_path_depth):
                os.makedirs(self.camera_path_depth)

            self.depth_log_file = open(os.path.join(self.output_path, "depth.txt"), "w")

            self.depth_log_file.write("# depth images \n" + \
                                      "# file: '%s' \n" % rosbag_name + \
                                      "# timestamp filename \n")

    def save_image_msg(self, msg, curr_frame_time):
        """
        Saves the ROS message as an PNG image
        :param msg: input ROS message for image
        :param curr_frame_time: current time in the bag file
        :return:
        """
        if self.idx % 2 == 0 and 'mono' in msg.encoding and RECORD_CAM_DEPTH:
            image_path = os.path.join(self.camera_path_depth, "%f.png" % curr_frame_time)
            opencv_image = CvBridge().imgmsg_to_cv2(msg, "mono8")
            cv2.imwrite(image_path, opencv_image)
        elif RECORD_CAM_RGB:
            image_path = os.path.join(self.camera_path_rgb, "%f.png" % curr_frame_time)
            # even numbered cameras record RGB images
            if self.idx % 2 == 0:
                opencv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            else:
                opencv_image = CvBridge().imgmsg_to_cv2(msg, "mono8")

            cv2.imwrite(image_path, opencv_image)

    def save_calibration_msg(self, msg):
        """
        Save calibration information form message into YAML file
        :param msg: input ROS message containing camera info
        :return:
        """
        calib_file_name = os.path.join(self.output_path, 'calib.yaml')

        # return if file exists (i.e. assume that data has been recorded once)
        if os.path.isfile(calib_file_name):
            return

        calib_file = open(calib_file_name, 'w')
        calib = {'image_width': msg.width,
                 'image_height': msg.height,
                 'camera_name': msg.header.frame_id,
                 'distortion_model': msg.distortion_model,
                 'distortion_coefficients': {'data': list(msg.D), 'rows': 1, 'cols': len(msg.D)},
                 'camera_matrix': {'data': list(msg.K), 'rows': 3, 'cols': 3},
                 'rectification_matrix': {'data': list(msg.R), 'rows': 3, 'cols': 3},
                 'projection_matrix': {'data': list(msg.P), 'rows': 3, 'cols': 4}}

        print(calib)
        yaml.safe_dump(calib, calib_file)

    def write_into_log_file(self, curr_frame_time):
        """
         Write the stored file name into the log files
        :param curr_frame_time: current time in the bag file
        :return:
        """
        # for rgb images
        if RECORD_CAM_RGB:
            self.rgb_log_file.write("%f rgb/%f.png\n" % (curr_frame_time, curr_frame_time))

        # for depth images
        if self.idx % 2 == 0 and RECORD_CAM_DEPTH:
            self.depth_log_file.write("%f depth/%f.png\n" % (curr_frame_time, curr_frame_time))


def extract_rosbag(rosbag_path, output_path):
    """
    Extract the RGBD data from the bag file
    :param rosbag_path: (str) path to the rosbag file
    :param output_path: (str) path to the directory to save the extracted data
    :return:
    """
    if not os.path.exists(rosbag_path) and rosbag_path[-4:] == ".bag":
        rospy.logerr("Invalid bagfile path: %s" % rosbag_path)
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Assemble desired topic names and generate mapping to camera indices.
    cameras = {}
    topics = []

    for camera_idx in CAM_NUM_IDXS:
        cameras['cam_%d' % camera_idx] = CameraIndexBag(camera_idx, output_path)
        cameras['cam_%d' % camera_idx].create_log_directories(rosbag_path)
        # append the camera topics associated into the list
        topics = topics + cameras['cam_%d' % camera_idx].topics

    print("Collecting topics: ", topics)

    # Data structure for the images.
    #   timestamp -> camera_idx -> image (message)
    frames = {}

    prev_frame_time = 0

    # Only keep images of a sliding window in memory.
    window_size = 5

    with rosbag.Bag(rosbag_path, "r") as bag:

        # Read messages and collect images with same timestamp.
        for topic, msg, t in bag.read_messages(topics=topics):
            # Allow aborting the script.
            if rospy.is_shutdown():
                return

            # Convert time object to number (in seconds) to be used as key.
            frame_time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9

            if frame_time not in frames:
                # If the timestamp occurs for the first frame_time,
                # create a data structure for its images.
                if frame_time <= prev_frame_time:
                    rospy.logerr("Received message out of window!")
                    return
                else:
                    frames[frame_time] = {}

            # extract camera index from topic name
            camera_idx = [int(idx) for idx in re.findall(r'\d+', topic)]
            if len(camera_idx) == 1:
                camera_idx = camera_idx[0]

            frames[frame_time][camera_idx] = msg

            if len(frames) > window_size:
                # Find the oldest frame and write it to disk if complete,
                # otherwise drop the frame.

                curr_frame_time = min(frames.keys())

                if curr_frame_time <= prev_frame_time:
                    rospy.logwarn("Wrong frame data ordering (%f after %f), dropping frame at timestamp %f seconds"
                                  % (curr_frame_time, prev_frame_time, curr_frame_time))
                    # Delete dropped frame from sliding window.
                    del frames[curr_frame_time]
                    continue

                prev_frame_time = curr_frame_time

                if len(frames[curr_frame_time]) != len(topics):
                    rospy.logwarn("Incomplete frame data at timestamps %f seconds (current: %d, expected: %d)"
                                  % (curr_frame_time, len(frames[curr_frame_time]), len(topics)))
                    # Delete dropped frame from sliding window.
                    del frames[curr_frame_time]
                    continue

                rospy.loginfo("Write images at timestamp: %f seconds" % curr_frame_time)

                # Write sensor messages into the system
                for cam_idx in frames[curr_frame_time]:
                    msg = frames[curr_frame_time][cam_idx]
                    msg_type = str(msg._type)
                    try:
                        if msg_type == "sensor_msgs/Image":
                            cameras['cam_%d' % cam_idx].save_image_msg(msg, curr_frame_time)
                        elif msg_type == "sensor_msgs/CameraInfo":
                            cameras['cam_%d' % cam_idx].save_calibration_msg(msg)
                        else:
                            rospy.logerr("Cannot save sensor messages from the topic: %s" % topic)
                            return

                        cameras['cam_%d' % cam_idx].write_into_log_file(curr_frame_time)

                    except CvBridgeError:
                        rospy.logerr(CvBridgeError)
                        return

                # Delete written frame from sliding window.
                del frames[curr_frame_time]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag_path", '-r', required=True)
    parser.add_argument("--output_path", '-o', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    rospy.init_node("rosbag_extractor", anonymous=True)
    try:
        extract_rosbag(args.rosbag_path, args.output_path)
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
