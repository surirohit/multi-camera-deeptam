import airsim
import math
import numpy as np
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


def printUsage():
    print("[ERROR] Usage: " + str(sys.argv[0] + " [input_pfm_file] [output_png_file]"))


if (len(sys.argv) != 3):
    printUsage()
    sys.exit(0)

depth_raw, scale = airsim.VehicleClient.read_pfm(sys.argv[1])
depth_raw = np.array(depth_raw) * 1000;
depth_vis = depth_raw.astype(np.uint16)
cv2.imwrite(sys.argv[2], depth_vis)
print("[INFO] Saved the output file: " + sys.argv[2])
