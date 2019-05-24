# Contributor: Mayank Mittal
# Script to collect data while the simulator is running

import airsim
import os
import math
import time
import numpy as np
import sys
import cv2
import pprint
import datetime

pp = pprint.PrettyPrinter(indent=4)

CLIPPING_MAX_DEPTH = 30.0   # in meters (set to zero for no clipping)
CLIPPING_MIN_DEPTH = 0.005   # in meters (set to zero for no clipping)
DEPTH_SCALE_FACTOR = 2000   # scaling factor for depth values

# help for usage
def printUsage():
    print("[ERROR] Usage: python airsim_collection.py")
    sys.exit()


# Convert degrees to radians
def toRadians(x):
    x = x * 1.0
    return x / 180 * math.pi


# convert coordinates from NED to ENU
def convert_ned_to_enu(pos_ned, orientation_ned):
    position_enu = airsim.Vector3r(pos_ned.x_val,
                                   - pos_ned.y_val,
                                   - pos_ned.z_val)
    orientation_enu = airsim.Quaternionr(orientation_ned.w_val,
                                         - orientation_ned.z_val,
                                         - orientation_ned.x_val,
                                         orientation_ned.y_val)

    pose_enu = airsim.Pose(position_enu, orientation_enu)
    return pose_enu


# add deafult txt for logFile for state
def addDefaultStateTxt(log_filename):
    default_txt = "# timestamp \t tx \t ty \t tz \t qx \t qy \t qz \t qw"
    with open(log_filename, 'a') as logFile:
        logFile.write("# Ground Truth Trajectory")
        logFile.write('\n')
        logFile.write("# Author: Mayank Mittal")
        logFile.write('\n')
        logFile.write(default_txt)
        logFile.write('\n')
    logFile.close()


# add default txt for logFile for images
def addDefaultImageTxt(log_filename, type_of_images="rgb"):
    default_txt = "# timestamp \t filename"
    with open(log_filename, 'a') as logFile:
        logFile.write("# " + type_of_images)
        logFile.write('\n')
        logFile.write("# Author: Mayank Mittal")
        logFile.write('\n')
        logFile.write(default_txt)
        logFile.write('\n')
    logFile.close()


# log the drone's state
# pose includes ground truth kinematics (i.e. position, orientation). All quantities are in NED coordinate system
def logState(simtime, log_filename, pose):
    pose_enu = convert_ned_to_enu(pose.position, pose.orientation)
    with open(log_filename, 'a') as logFile:
        logFile.write(str(simtime) + '\t')
        logFile.write(str(pose_enu.position.x_val) + '\t')
        logFile.write(str(pose_enu.position.y_val) + '\t')
        logFile.write(str(pose_enu.position.z_val) + '\t')
        logFile.write(str(pose_enu.orientation.x_val) + '\t')
        logFile.write(str(pose_enu.orientation.y_val) + '\t')
        logFile.write(str(pose_enu.orientation.z_val) + '\t')
        logFile.write(str(pose_enu.orientation.w_val) + '\t')
        logFile.write('\n')
    logFile.close()


# log the collected images
def logImage(simtime, log_filename, img_filename):
    with open(log_filename, 'a') as logFile:
        logFile.write(str(simtime) + '\t')
        logFile.write(str(img_filename))
        logFile.write('\n')
    logFile.close()


# convert float64 image to 16UC1 image (following OpenNI representation)
def save_depth_float_as_uchar16(filename, depth_array):
    if CLIPPING_MAX_DEPTH > 0:
        depth_array[depth_array > CLIPPING_MAX_DEPTH] = 0
    if CLIPPING_MIN_DEPTH > 0:
        depth_array[depth_array < CLIPPING_MIN_DEPTH] = 0

    depth_vis = np.array(depth_array) * DEPTH_SCALE_FACTOR
    depth_vis = depth_vis.astype(np.uint16)
    cv2.imwrite(filename, depth_vis)


# convert float64 image to heatmap (following OpenNI representation)
def save_depth_float_as_heatmap(filename, depth_raw):
    # depth_raw = MultirotorClient.getPfmArray(response)
    depth_vis = np.array(depth_raw);
    normalized_im = (depth_vis - np.min(depth_vis)) / np.max(depth_vis)
    normalized_im = 1 - normalized_im
    depth_vis_heatmap = cv2.applyColorMap(np.uint8(255 * normalized_im), cv2.COLORMAP_JET)
    cv2.imwrite(filename, depth_vis_heatmap)


# setting up directories if they do not exist already
def setupDirectories():
    global PATH

    # directory path names
    leftCamRgbPath = PATH + os.path.normpath('/left_cam/rgb') + "/"
    rightCamRgbPath = PATH + os.path.normpath('/right_cam/rgb') + "/"
    leftCamDepthPath = PATH + os.path.normpath('/left_cam/depth') + "/"
    rightCamDepthPath = PATH + os.path.normpath('/right_cam/depth') + "/"

    ## ensure that the order here and in retrieving response call is the same
    dataPath = [leftCamRgbPath, rightCamRgbPath, leftCamDepthPath, rightCamDepthPath]

    # create directories if they do not exist
    for path in dataPath:
        if not os.path.exists(path):
            os.makedirs(path)

    print("[INFO] Directories setup successful at: " + PATH)
    return dataPath


# capture images through AirSim APIs
def captureImages(dataPath, ctr, filename):
    # Check if the file already exists or not
    # If it exists then skip the retrieval of responses from the simulator
    # NOTE: Should only be used only when dataset collection has to be resumed

    # flag = 0 means that the files are not present/missing, hence data has to be recorded
    flag = 0
    for i in range(0, len(dataPath)):
        filepath = dataPath[i] + '/' + filename
        if (os.path.isfile(os.path.normpath(filepath + '.png')) or os.path.isfile(os.path.normpath(filepath + '.pfm'))):
            flag = 1
            print("File exists")
            break
        else:
            flag = 0

    if (flag == 0):
        # Retrive responses from AirSim client
        responses = client.simGetImages([
            airsim.ImageRequest("front_left", airsim.ImageType.Scene),
            airsim.ImageRequest("front_right", airsim.ImageType.Scene),
            airsim.ImageRequest("front_left", airsim.ImageType.DepthPlanner, True),
            airsim.ImageRequest("front_right", airsim.ImageType.DepthPlanner, True)])

        for i, response in enumerate(responses):
            if response.pixels_as_float:
                depth_raw = airsim.get_pfm_array(response)
                save_depth_float_as_uchar16(os.path.normpath(dataPath[i] + '/' + filename + '.png'), depth_raw)
            else:
                airsim.write_file(os.path.normpath(dataPath[i] + '/' + filename + '.png'), response.image_data_uint8)


######################@#### PATH Control Variables ######@#####################
INTERVAL = 0.2  # duration between two successive images
PATH = os.path.dirname(os.path.abspath(__file__)) + os.path.normpath('/airsim_dataset/')  # Folder path to save data
##############################################################################

from_date = datetime.datetime.strptime("2018_04_01", "%Y_%m_%d").strftime("%Y_%m_%d")
PATH = PATH + "/" + str(from_date)

# log file name
if not os.path.exists(PATH):
    os.makedirs(PATH)
# setup the directories to store the dataset
dataPath = setupDirectories()

# add default header texts
log_filename = PATH + "/groundtruth.txt"
left_cam_log_filename = PATH + "/left_cam//groundtruth.txt"
right_cam_log_filename = PATH + "/right_cam/groundtruth.txt"
left_cam_rgb_filename = PATH + "/left_cam/rgb.txt"
left_cam_depth_filename = PATH + "/left_cam/depth.txt"
right_cam_rgb_filename = PATH + "/right_cam/rgb.txt"
right_cam_depth_filename = PATH + "/right_cam/depth.txt"

# add default texts
addDefaultStateTxt(log_filename);
addDefaultImageTxt(left_cam_rgb_filename, "left_cam_rgb")
addDefaultImageTxt(right_cam_rgb_filename, "right_cam_rgb")
addDefaultImageTxt(left_cam_depth_filename, "left_cam_depth")
addDefaultImageTxt(right_cam_depth_filename, "left_cam_depth")

# Initiate image number counter
ctr = 1

# Connect to the simulator!
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
print("Connection successful to the drone!")

# Set default pose :)
# client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, H), airsim.to_quaternion(0, 0, 0)), True)

# change orientation of cameras
# client.simSetCameraOrientation("front_left", airsim.to_quaternion(0, 0, 0))
# client.simSetCameraOrientation("front_right", airsim.to_quaternion(0, 0, toRadians(36)))

for camera_name in ["front_left", "front_right"]:
    camera_info = client.simGetCameraInfo(str(camera_name))
    print("CameraInfo %s: " % (camera_name))
    pp.pprint(camera_info)

print("--- Starting Dataset Collection! ---")
airsim.wait_key('Hit Enter to start')

# fly the drone at various poses with height = z
while True:
    # define filename
    sim_time = time.time()
    img_filename = str(sim_time)

    # pause for specified interval (to ensure soft framerate)
    print("Capturing Responses from Simulator: Count " + str(ctr))
    print("Filename: " + img_filename)
    time.sleep(INTERVAL)

    # get pose from simulator
    pose = client.simGetVehiclePose()
    # capture images using AirSim API
    captureImages(dataPath, ctr, img_filename)

    # log the drone state into a file
    logState(sim_time, log_filename, pose)
    logImage(sim_time, left_cam_rgb_filename, 'rgb/' + img_filename + ".png")
    logImage(sim_time, right_cam_rgb_filename, 'rgb/' + img_filename + ".png")
    logImage(sim_time, left_cam_depth_filename, 'depth/' + img_filename + ".png")
    logImage(sim_time, right_cam_depth_filename, 'depth/' + img_filename + ".png")

    # log the camera's state
    logState(sim_time, left_cam_log_filename, client.simGetCameraInfo("front_left").pose)
    logState(sim_time, right_cam_log_filename, client.simGetCameraInfo("front_right").pose)

    ctr = ctr + 1
    print('-------------------------')

print("Dataset collected!")
