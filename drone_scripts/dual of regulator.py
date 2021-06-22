#from dronekit import connect
#from pymavlink import mavutil
#import dronekit as dk
import asyncio
import time
import airsim
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import re

OSETFWD = 3
OSETSIDE = 0.5
CAM_FOV = 100
IMG_WIDTH = 80
IMG_HEIGHT = 42
CAM_ANGLE = 75
SACCADE_COOLDOWN = 3
PPD = IMG_WIDTH / CAM_FOV
SACCADE_TRIGGER = 0.9

DISFlow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

# (X=1960.000000,Y=-6250.000000,Z=242.000000)
# (X=1960.000000,Y=-6540.000000,Z=242.000000)
# (X=1960.000000,Y=-6880.000000,Z=242.000000)
# (X=1960.000000,Y=-7190.000000,Z=242.000000)

start = "(X=1960.000000,Y=-6250.000000,Z=242.000000)"#"(X=-1770.000000,Y=10570.000000,Z=0.000000)"
startFormed = re.findall("['X', 'Y']=-*[0-9]*", start)
xStart = float(startFormed[0][2:])/100
yStart = float(startFormed[1][2:])/100


client = airsim.MultirotorClient()
client.confirmConnection()

def relPosToAbs(x, y, z):
    rotMat = R.from_quat(client.simGetVehiclePose().orientation.to_numpy_array()).as_matrix()
    rotated = np.matmul(rotMat, np.array([[x], [y], [z]]))
    #print(rotated)
    return rotated[0][0], rotated[1][0], rotated[2][0]

def angleForwardToWorld(angleDif, vel):
    pitch, roll, yaw  = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    vx = vel * math.cos(yaw + angleDif)
    vy = vel * math.sin(yaw + angleDif)
    return vx, vy

def getImage(cam):
    rawImage = client.simGetImage(cam, airsim.ImageType.Scene)
    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
    return png

def sideControl(lOF, rOF):
    maxFlow = max(lOF, rOF)
    print(maxFlow)
    flowError = OSETSIDE - maxFlow
    control = flowError if rOF > lOF else -flowError
    control = control * 0.3
    return control

def fwdControl(sumOF):
    control = OSETFWD - sumOF
    return control

def getImageAtAngle(wL, wR, angle, spread):
    startPix = int((CAM_FOV/2 - CAM_ANGLE + angle - spread/2) * PPD)
    endPix = startPix + int(spread * PPD)
    newWR = wR[:,startPix:endPix]
    newWL = wL[:,IMG_WIDTH - endPix:IMG_WIDTH - startPix]
    return newWL, newWR

def getAngleOfInc(wL, wR):
    i = min(abs(math.atan(1.732 * (wL - wR) / (wL + wR))), math.pi/4)
    return i

def saccadeGen(wL30, wR30):
    i = getAngleOfInc(wL30, wR30)
    print(i)
    return math.pi/2 - i if wL30 > wR30 else i - math.pi/2

def imFromResp(resp):
    data = airsim.string_to_uint8_array(resp.image_data_uint8)
    return data.reshape(resp.height, resp.width, 3)

def getImages():
    responses = client.simGetImages([
        airsim.ImageRequest("main_left", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("main_right", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("main_front", airsim.ImageType.Scene, False, False),
        #airsim.ImageRequest("main_back", airsim.ImageType.Scene, False, False)
    ])
    left = imFromResp(responses[0])
    right = imFromResp(responses[1])
    front = imFromResp(responses[2])
    #emptyArray[:, :IMG_WIDTH] = self.imFromResp(responses[0])
    #emptyArray[:, IMG_WIDTH:] = self.imFromResp(responses[1])
    return left, right, front

def formatImg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')  # , (3, 3))
    img[0,0] = 0 # makes the background black
    img = np.take(img, self.mapping)
    return img

def _angleDiff(a1, a2):
    return (((a1 - a2) + 180) % 360) - 180

def main():
    global IMG_WIDTH
    global IMG_HEIGHT
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    landed = client.getMultirotorState().landed_state
    if landed == airsim.LandedState.Landed:
        print("taking off...")
        client.takeoffAsync()
    else:
        print("already flying...")
        client.hoverAsync().join()

    print("Getting images")

    """rawL = getImage("main_left")
    rawR = getImage("main_right")
    rawF = getImage("front")"""
    rawL, rawR, rawF = getImages()

    IMG_HEIGHT, IMG_WIDTH, channels = rawL.shape
    prevL = cv2.cvtColor(rawL, cv2.COLOR_BGR2GRAY)
    prevR = cv2.cvtColor(rawR, cv2.COLOR_BGR2GRAY)
    prevF = cv2.cvtColor(rawF, cv2.COLOR_BGR2GRAY)

    flowL = np.zeros((prevL.shape[0], prevL.shape[1], 2))
    flowR = np.zeros((prevR.shape[0], prevR.shape[1], 2))
    flowF = np.zeros((prevF.shape[0], prevF.shape[1], 2))
    xPoses = []
    yPoses = []

    fVel = 2.5
    prevCon = 0
    prevWL = 0
    prevWR = 0
    lastSac = 0
    _, _, targetYaw = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    counter = 0
    travelArray = []
    while counter < 1000:
        pose = client.simGetVehiclePose().position
        if counter % 10 == 0:
            travelArray.append([[xStart + pose.x_val], [yStart + pose.y_val]])
        if (counter > 800):
            t = np.array(travelArray)
            np.save("eval" + "_0", t)
            exit()
        counter += 1
        #t0 = time.time()
        """rawL = getImages("main_left")
        rawR = getImages("main_right")
        rawF = getImages("front")"""
        rawL, rawR, rawF = getImages()

        greyL = cv2.cvtColor(rawL, cv2.COLOR_BGR2GRAY)#, (3, 3))
        greyR = cv2.cvtColor(rawR, cv2.COLOR_BGR2GRAY)#, (3, 3))
        greyF = cv2.cvtColor(rawF, cv2.COLOR_BGR2GRAY)#, (3, 3))
        cv2.imshow('frame123', cv2.resize(greyF, (0, 0), fx=4, fy=4))
        flowL = DISFlow.calc(prevL, greyL, flowL)
        flowR = DISFlow.calc(prevR, greyR, flowR) #cv2.calcOpticalFlowFarneback(prevR, greyR, None, 0.5, 3, 19, 3, 9, 1.7, 0)
        flowF = DISFlow.calc(prevF, greyF, flowF)

        prevL = greyL
        prevR = greyR
        prevF = greyF

        flow15Left, flow15Right = flowF[:,0:10], flowF[:,30:] # 0 to 20 deg
        flow30Left, flow30Right = getImageAtAngle(flowL, flowR, 30, 10) # 25 to 35 deg
        # I SHOULD CHECK THE SIGN USING 45 DEGREE ANGLE AS THIS WILL MOVE TO THE AREA WITH THE LEAST OPTICAL FLOW (MOST FREE SPACE)
        flow80Left, flow80Right = getImageAtAngle(flowL, flowR, 80, 45) # 35 to 125 deg

        wL15 = abs(np.mean(flow15Left[:, :, 0])) * 0.15 + 0.85 * prevWL
        wR15 = abs(np.mean(flow15Right[:, :, 0])) * 0.15 + 0.85 * prevWR
        wL30 = abs(np.mean(flow30Left[:, :, 0]))
        wR30 = abs(np.mean(flow30Right[:, :, 0]))
        wL80 = abs(np.mean(flow80Left[:, :, 0]))
        wR80 = abs(np.mean(flow80Right[:, :, 0]))
        #print(wL15 + wR15)
        curPitch, curRoll, curYaw = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
        if wL15 + wR15 > SACCADE_TRIGGER and (time.time() - lastSac) > SACCADE_COOLDOWN:
            localTurn = saccadeGen(wL30, wR30)
            targetYaw = curYaw+localTurn
            lastSac = time.time()
            print(curYaw)
            print("turnt ", localTurn)

        sC = sideControl(wL80, wR80)
        fC = fwdControl(wL80 + wR80) * 0.2 + 0.8 * prevCon
        prevCon = fC # Smooth out this value as it's a bit all over the place
        moveX, moveY, moveZ = relPosToAbs(fC, sC, 0)
        #print(moveX, moveY, moveZ)
        client.moveByVelocityZAsync(moveX, moveY, -1.5, 0.07, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, _angleDiff(targetYaw, curYaw) * 70))

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('x'):
            break

        prevWL = wL15
        prevWR = wR15
        #t1 = time.time()
        #print(t1 - t0)
    print(xPoses)
    print(yPoses)
main()