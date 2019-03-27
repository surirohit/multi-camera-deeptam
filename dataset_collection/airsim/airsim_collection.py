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

# help for usage
def printUsage():
    print("[ERROR] Usage: python airsim_collection.py")
    sys.exit()

# Convert degrees to radians
def toRadians(x):
     x = x * 1.0
     return x/180*math.pi

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
def addDefaultImageTxt(log_filename, type_of_images = "rgb"):
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
    with open(log_filename, 'a') as logFile:
        logFile.write(str(simtime) + '\t')
        logFile.write(str(pose.position.x_val) + '\t')
        logFile.write(str(pose.position.y_val) + '\t')
        logFile.write(str(pose.position.z_val) + '\t')
        logFile.write(str(pose.orientation.x_val) + '\t')
        logFile.write(str(pose.orientation.y_val) + '\t')
        logFile.write(str(pose.orientation.z_val) + '\t')
        logFile.write(str(pose.orientation.w_val) + '\t')
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
def save_depth_float_as_uchar16(filename, depth_raw):
     # depth_raw = MultirotorClient.getPfmArray(response)
     depth_vis = np.array(depth_raw) * 1000;
     depth_vis = depth_vis.astype(np.uint16)
     cv2.imwrite(filename, depth_vis)

# convert float64 image to heatmap (following OpenNI representation)
def save_depth_float_as_heatmap(filename, depth_raw):
     # depth_raw = MultirotorClient.getPfmArray(response)
     depth_vis = np.array(depth_raw);
     normalized_im = (depth_vis - np.min(depth_vis))/np.max(depth_vis)
     normalized_im = 1 - normalized_im
     depth_vis_heatmap = cv2.applyColorMap(np.uint8(255*normalized_im), cv2.COLORMAP_JET)
     cv2.imwrite(filename, depth_vis_heatmap)

# setting up directories if they do not exist already
def setupDirectories():
    global PATH

    # directory path names
    leftCamRgbPath = PATH + os.path.normpath('/left_cam_rgb') + "/"
    rightCamRgbPath = PATH + os.path.normpath('/right_cam_rgb') + "/"
    leftCamDepthPath = PATH + os.path.normpath('/left_cam_depth_vis') + "/"
    rightCamDepthPath = PATH + os.path.normpath('/right_cam_depth_vis') + "/"

    ## ensure that the order here and in retrieving response call is the same
    dataPath = [leftCamRgbPath, rightCamRgbPath, leftCamDepthPath, rightCamDepthPath]

    #create directories if they do not exist
    for path in dataPath:
        if not os.path.exists(path) :
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
                airsim.write_pfm(os.path.normpath(dataPath[i] + '/' + filename + '.pfm'), airsim.get_pfm_array(response))
            else:
                airsim.write_file(os.path.normpath(dataPath[i] + '/' + filename + '.png'), response.image_data_uint8)


######################@#### PATH Control Variables ######@#####################
INTERVAL = 0.2        # duration between two successive images
PATH = os.path.dirname(os.path.abspath(__file__)) + os.path.normpath('/airsim_dataset/') # Folder path to save data
H = -10               # provide inital height
##############################################################################

from_date = datetime.datetime.strptime("2018_04_01", "%Y_%m_%d").strftime("%Y_%m_%d")
PATH = PATH + "/" + str(from_date)

# log file name
if not os.path.exists(PATH) :
    os.makedirs(PATH)
log_filename = PATH + "/groundtruth.txt"
left_cam_rgb_filename = PATH + "/left_cam_rgb.txt"
left_cam_depth_filename = PATH + "/left_cam_depth.txt"
right_cam_rgb_filename = PATH + "/right_cam_rgb.txt"
right_cam_depth_filename = PATH + "/right_cam_depth.txt"

# add default texts
addDefaultStateTxt(log_filename);
addDefaultImageTxt(left_cam_rgb_filename, "left_cam_rgb")
addDefaultImageTxt(right_cam_rgb_filename, "right_cam_rgb")
addDefaultImageTxt(left_cam_depth_filename, "left_cam_depth")
addDefaultImageTxt(right_cam_depth_filename, "left_cam_depth")

# setup the directories to store the dataset
dataPath = setupDirectories()

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

for camera_name in range(5):
    camera_info = client.simGetCameraInfo(str(camera_name))
    print("CameraInfo %d: %s" % (camera_name, pp.pprint(camera_info)))

print("--- Starting Dataset Collection! ---")
airsim.wait_key('Hit Enter to start')


# fly the drone at various poses with height = z
while True:
    # define filename
    sim_time = time.time()
    img_filename = str(sim_time)
    ctr = ctr + 1

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
    logImage(sim_time, left_cam_rgb_filename, dataPath[0] + img_filename)
    logImage(sim_time, right_cam_rgb_filename, dataPath[1] + img_filename)
    logImage(sim_time, left_cam_depth_filename, dataPath[2] + img_filename)
    logImage(sim_time, right_cam_depth_filename, dataPath[3] + img_filename)

    print('-------------------------')

print("Dataset collected!")
