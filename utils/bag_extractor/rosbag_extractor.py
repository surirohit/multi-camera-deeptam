#!/usr/bin/python

import os
import sys
import argparse

import rosbag
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Change only this
cam_nums = [0,2,4,6]

def extract_rosbag(rosbag_path, output_path):
    if not os.path.exists(rosbag_path) and rosbag_path[-4:] == ".bag":
        rospy.logerr("Invalid bagfile path")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Assemble desired topic names and generate mapping to camera indices.
    topics = []
    idx_files = {}
    image_topic_to_camera_idx = {}

    for camera_idx in cam_nums:
        image_topic = "/uvc_camera/cam_%d/image_raw" % (camera_idx)
        topics.append(image_topic)
        image_topic_to_camera_idx[image_topic] = camera_idx
        image_topic = "/uvc_camera/cam_%d/image_depth" % (camera_idx)
        topics.append(image_topic)
        image_topic_to_camera_idx[image_topic] = camera_idx + 1

        camera_path_rgb = os.path.join(output_path, \
            "cam_%d" % camera_idx, 'rgb')
        if not os.path.exists(camera_path_rgb):
            os.makedirs(camera_path_rgb)

        camera_path_depth = os.path.join(output_path, \
            "cam_%d" % camera_idx, 'depth')
        if not os.path.exists(camera_path_depth):
            os.makedirs(camera_path_depth)

        idx_files["cam_%d" % camera_idx] = \
            open(os.path.join(output_path, \
                "cam_%d" % camera_idx, "rgb.txt"), "w")
        idx_files["cam_%d" % (camera_idx+1)] = \
            open(os.path.join(output_path, \
                "cam_%d" % camera_idx, "depth.txt"), "w")

        idx_files["cam_%d" % camera_idx].write("# color images \n" + \
                        "# file: '%s' \n" % rosbag_path + \
                        "# timestamp filename \n")
        idx_files["cam_%d" % (camera_idx+1)].write("# depth images \n" + \
                        "# file: '%s' \n" % rosbag_path + \
                        "# timestamp filename \n")

    # Data structure for the images.
    #   timestamp -> camera_idx -> image (message)
    frames = {}

    # CvBridge to convert ROS images to OpenCV images to save them.
    opencv_bridge = CvBridge()

    prev_frame_time = 0

    # Only keep images of a sliding window in memory.
    window_size = 5

    with rosbag.Bag(rosbag_path, "r") as bag:

        # Read messages and collect images with same timestamp.
        for topic, msg, t in bag.read_messages(topics=topics):
            # Allow aborting the script.
            if rospy.is_shutdown():
                return

            # Convert time object to number to be used as key.
            frame_time = int(msg.header.stamp.secs * 1e9
                                + msg.header.stamp.nsecs)

            if frame_time not in frames:
                # If the timestamp occurs for the first frame_time,
                # create a data structure for its images.
                if frame_time <= prev_frame_time:
                    rospy.logerr("Received message out of window")
                    return
                else:
                    frames[frame_time] = {}

            camera_idx = image_topic_to_camera_idx[topic]
            frames[frame_time][camera_idx] = msg

            if len(frames) > window_size:
                # Find the oldest frame and write it to disk if complete,
                # otherwise drop the frame.

                curr_frame_time = min(frames.keys())

                if curr_frame_time <= prev_frame_time:
                    rospy.logwarn(
                        "Wrong frame ordering (%f after %f), "
                        "dropping frame at time %f"
                        % (curr_frame_time, prev_frame_time, curr_frame_time))
                    # Delete dropped frame from sliding window.
                    del frames[curr_frame_time]
                    continue

                prev_frame_time = curr_frame_time

                if len(frames[curr_frame_time]) != len(topics):
                    rospy.logwarn(
                        "Incomplete frame at time %f (%d images)"
                        % (curr_frame_time, len(frames[curr_frame_time])))
                    # Delete dropped frame from sliding window.
                    del frames[curr_frame_time]
                    continue

                rospy.loginfo("Write images at time %d" % curr_frame_time)

                # Write timestamp to file.
                for cam in frames[curr_frame_time]:
                    msg = frames[curr_frame_time][cam]
                    try:
                        image_path = None
                        if cam%2 == 0:
                            opencv_image = opencv_bridge.imgmsg_to_cv2(msg, "bgr8")
                            image_path = "%s/cam_%d/rgb/%d.png" \
                                        % (output_path, cam, curr_frame_time)
                        else:
                            opencv_image = opencv_bridge.imgmsg_to_cv2(msg, "mono8")
                            image_path = "%s/cam_%d/depth/%d.png" \
                                        % (output_path, cam-1, curr_frame_time)

                        cv2.imwrite(image_path, opencv_image)
                        
                        if cam%2==0:
                            idx_files["cam_%d" % cam].write("%d rgb/%d.png\n"\
                                %(curr_frame_time,curr_frame_time))
                        else:
                            idx_files["cam_%d" % cam].write("%d depth/%d.png\n"\
                                %(curr_frame_time,curr_frame_time))


                    except CvBridgeError, e:
                        rospy.logerr(e)
                        return

                # Delete written frame from sliding window.
                del frames[curr_frame_time]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag_path", required=True)
    parser.add_argument("--output_path", required=True)
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
