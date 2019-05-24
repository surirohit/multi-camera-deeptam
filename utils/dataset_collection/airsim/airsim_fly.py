## Contributor: Mayank Mittal
# Script to fly the spawned drone in lawn mower patter

'''
~ #       #########
^ #       #       #
| #       #       #
| #       #       #
Y #       #       #
| #       #       #
| #       #       #
v #       #       #
~ #########       #
  <-X_step->
'''

import airsim
import sys
import time
import math

########################@#### Control Variables ######@#######################
# Note: In UnrealEngine the coordinate system is inverted
X = 10  # final x coordinates
Y = -25  # length of path traversed along x
# The paramters H and X_step need to changed to collect more data
H = -2.5  # height of flight
X_step = 10.0

# defining variables for flight
YAW = 0  # yaw of the drone while flying (set to 0 assuming)
V = 0.5  # speed of drone while flying


##############################################################################

# color class to prettify the terminal outputs being printed
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# print in green
def printg(message=''):
    if message != '':
        print(bcolors.OKGREEN + message + bcolors.ENDC)


# Connect to the simulator!
client = airsim.MultirotorClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
print("Connection successful to the drone!")

# arming the drone
if (client.isApiControlEnabled()):
    if (client.armDisarm(True)):
        print(bcolors.OKBLUE + "drone is armed" + bcolors.ENDC)
else:
    print(bcolors.FAIL + "failed to arm the drone" + bcolors.ENDC)
    sys.exit(1);

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print(bcolors.OKBLUE + "drone should now be flying..." + bcolors.ENDC)
    client.takeoffAsync().join()
else:
    print(bcolors.WARNING + "it appears the drone is already flying" + bcolors.ENDC)
    print(bcolors.WARNING + "kindly restart the simulator to ensure proper flying" + bcolors.ENDC)
    client.hoverAsync().join()

# to pause the drone for a while to stabilize the flying altitude
client.moveToPositionAsync(0, 0, H, V, 60, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, YAW), -1,
                           0).join()
print(bcolors.OKBLUE + "moved to altitude " + str(H) + bcolors.ENDC)

x = 0  # initial x coordinates
y = Y
delay_x = abs(X_step / V)  # delay for path along y
delay_y = abs(Y / V)  # delay for path along x

while x < X:
    printg("starting to move to (%3d, %3d)" % (x, y))
    client.moveToPositionAsync(x, y, H, V, delay_y, airsim.DrivetrainType.MaxDegreeOfFreedom,
                               airsim.YawMode(False, YAW), -1, 0).join()

    x = x + X_step

    printg("starting to move to (%3d, %3d)" % (x, y))
    client.moveToPositionAsync(x, y, H, V, delay_x, airsim.DrivetrainType.MaxDegreeOfFreedom,
                               airsim.YawMode(False, YAW), -1, 0).join()

    printg("starting to move to (%3d, %3d)" % (x, y))
    client.moveToPositionAsync(x, 0, H, V, delay_y, airsim.DrivetrainType.MaxDegreeOfFreedom,
                               airsim.YawMode(False, YAW), -1, 0).join()

    x = x + X_step

    printg("starting to move to (%3d, %3d)" % (x, y))
    client.moveToPositionAsync(x, 0, H, V, delay_x, airsim.DrivetrainType.MaxDegreeOfFreedom,
                               airsim.YawMode(False, YAW), -1, 0).join()

# to record the time of flight of the drone
client.landAsync(10)
