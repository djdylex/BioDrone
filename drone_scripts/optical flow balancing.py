#from dronekit import connect
#from pymavlink import mavutil
#import dronekit as dk
import asyncio
import time
import airsim
import cv2
import math
import numpy as np

def angleForwardToWorld(client, angleDif, vel):
    pitch, roll, yaw  = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    vx = 0#vel #* math.cos(yaw + angleDif)
    vy = -vel #* math.sin(yaw + angleDif)
    return vx, vy

client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
time.sleep(0.5)
client.enableApiControl(True)
client.armDisarm(True)

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()
else:
    print("already flying...")
    client.hoverAsync().join()


print("Getting images")
rawImage = client.simGetImage("main_front", airsim.ImageType.Scene)
png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
imgHeight, imgWidth, channels = png.shape
prev = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
fVel = 0.5
while True:
    rawImage = client.simGetImage("main_left", airsim.ImageType.Scene)
    grey = ""
    if (rawImage == None):
        print("Camera is not returning image, please check airsim for error messages")
        sys.exit(0)
    else: 
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        grey = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Scene", png)

    flow = cv2.calcOpticalFlowFarneback(prev,grey,None,0.5,3,25,3,13,2,0)
    prev = grey

    avgLeft = 0
    avgRight = 0
    for y in range(0, int(imgHeight/2)):
        for x in range(0, int(imgWidth/2)):
            avgLeft += flow[y, x, 0]
            avgRight += flow[y, x + int(imgWidth/2), 0]

    avgLeft = abs(avgLeft/(imgWidth*imgHeight/2))
    avgRight = abs(avgRight/(imgWidth*imgHeight/2))
    print((avgLeft, avgRight))
    control = avgLeft - avgRight # Positive if bigger flow on right
    vx, vy = angleForwardToWorld(client, control/10, fVel)
    client.moveByVelocityZAsync(vx, vy, -1.5, 0.05, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0))
    
    
    key = cv2.waitKey(1) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x')):
        break
