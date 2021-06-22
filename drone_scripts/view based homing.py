#from dronekit import connect
#from pymavlink import mavutil
#import dronekit as dk
import asyncio
import time
import airsim
import cv2
import math
import numpy as np
import keyboard
from scipy.spatial.transform import Rotation as R
from enum import Enum

OSETFWD = 2
OSETSIDE = 0.75
CAM_FOV = 120 # HOR FOV
IMG_WIDTH = 480
IMG_HEIGHT = 270
SACCADE_COOLDOWN = 3
DEG_RES = 2 # How many pixels per degree resolution
SACCADE_TRIGGER = 0.8
ms = Enum('moveStates', ['sac', 'intSac']) # movement States
ps = Enum('actStates', ['search', 'home', 'idle']) # passive states

sinVals = []
cosVals = []
tanVals = []
for i in range(-int(CAM_FOV / 2 * DEG_RES), int(CAM_FOV / 2 * DEG_RES)):
    sinVals.append(math.sin(math.radians(i / DEG_RES)))
    cosVals.append(math.cos(math.radians(i / DEG_RES)))
    tanVals.append(1 - math.tan(math.radians(abs(i / DEG_RES)) - math.pi/2))

DISFlow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

client = airsim.MultirotorClient()
client.confirmConnection()


class BioDrone:
    def __init__(self):
        self.turnRate = 90 # deg/sec
        self.fVel = 1.5
        self.state = ms.intSac
        self.turnTrig = 50
        self.targetYaw = 0
        self.threshold = 60
        self.gain = 3
        self.goalDir = 0 # Direction of goal to move in, default 0 for forward, used also for homing
        self.cdLength = 1.2 # Cooldown after turn in secs

    @staticmethod
    def _clampRot(rot): # Keeps to convention of +- 180 from front
        return ((rot + 180) % 360) - 180

    def relPosToAbs(self, x, y, z):
        rotMat = R.from_quat(client.simGetVehiclePose().orientation.to_numpy_array()).as_matrix()
        rotated = np.matmul(rotMat, np.array([[x], [y], [z]]))
        #print(rotated)
        return rotated[0][0], rotated[1][0], rotated[2][0]

    def calcMapping(self): # Inverse gnomonic projection, then equirectangular projection
        outWidth = round(CAM_FOV * DEG_RES)
        outHeight = round(outWidth * IMG_HEIGHT/IMG_WIDTH)
        focalLength = (IMG_WIDTH / 2) / math.tan(math.radians(CAM_FOV) / 2)  # in pixel units

        mapping = np.zeros((outHeight, outWidth), dtype=np.int32)

        mpx = (IMG_WIDTH - 1)/2
        mpy = (IMG_HEIGHT - 1)/2
        outMpx = (outWidth - 1)/2
        outMpy = (outHeight - 1)/2

        for outX in range(outWidth):
            az = (outX - outMpx) / DEG_RES
            x = focalLength * math.tan(math.radians(az))
            tempX = round(x + mpx)
            for outY in range(outHeight):
                el = (outY - outMpy) / DEG_RES
                y = round(math.sqrt(focalLength ** 2 + x **2) * math.tan(math.radians(el)) + mpy)
                if 0 <= y < IMG_HEIGHT and 0 <= tempX < IMG_WIDTH:
                    mapping[outY, outX] = (y * IMG_WIDTH) + tempX #[y + mpy, x + mpx]

        return mapping

    def angleForwardToWorld(self, angleDif, vel):
        pitch, roll, yaw  = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
        vx = vel * math.cos(yaw + angleDif)
        vy = vel * math.sin(yaw + angleDif)
        return vx, vy

    def getImage(self, cam):
        rawImage = client.simGetImage(cam, airsim.ImageType.Scene)
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        return png

    def imFromResp(self, resp):
        data = airsim.string_to_uint8_array(resp.image_data_uint8)
        return data.reshape(resp.height, resp.width, 3)

    def getImages(self):
        responses = client.simGetImages([
            #airsim.ImageRequest("main_left", airsim.ImageType.Scene, False, False),
            #airsim.ImageRequest("main_right", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("main_front", airsim.ImageType.Scene, False, False),
            #airsim.ImageRequest("main_back", airsim.ImageType.Scene, False, False)
        ])
        left = 0#self.imFromResp(responses[0])
        right = 0#self.imFromResp(responses[1])
        front = self.imFromResp(responses[0])
        back = 0#self.imFromResp(responses[3])
        return left, right, front, back

    def initSeq(self):
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

    def main(self):
        global IMG_WIDTH
        global IMG_HEIGHT

        self.initSeq()

        mapping = self.calcMapping()

        print("Getting images")

        rawL, rawR, rawF, rawB = self.getImages()

        #greyL = cv2.cvtColor(rawL, cv2.COLOR_BGR2GRAY)  # , (3, 3))
        #greyR = cv2.cvtColor(rawR, cv2.COLOR_BGR2GRAY)  # , (3, 3))
        greyF = cv2.cvtColor(rawF, cv2.COLOR_BGR2GRAY)  # , (3, 3))
        greyF = greyF.astype('uint8')
        #greyF[0,0] = 0
        greyF = np.take(greyF, mapping)

        homeVector = [0.01, 0.01]

        homeMode = False
        hsv = np.zeros_like(rawF)
        hsv[..., 1] = 255
        detection = np.zeros((greyF.shape[1]))
        flow = np.zeros((greyF.shape[0], greyF.shape[1], 2))

        cooldown = time.time()

        counter = 0
        travelArray = []
        while True:
            side = 0
            conX = 0
            conRot = 0
            t0 = time.time()
            rawL, rawR, rawF, rawB = self.getImages()
            #print(rawL.shape)

            #prevL = greyL
            #prevR = greyR
            prevF = greyF

            """greyL = cv2.cvtColor(rawL, cv2.COLOR_BGR2GRAY)#, (3, 3))
            greyL = greyL.astype('uint8')
            greyL = np.take(greyL, mapping)
            greyR = cv2.cvtColor(rawR, cv2.COLOR_BGR2GRAY)#, (3, 3))
            greyR = greyR.astype('uint8')
            greyR = np.take(greyR, mapping)
            greyB = cv2.cvtColor(rawB, cv2.COLOR_BGR2GRAY)
            greyB = greyB.astype('uint8')
            greyB = np.take(greyB, mapping)"""
            greyF = cv2.cvtColor(rawF, cv2.COLOR_BGR2GRAY)
            greyF[0,0] = 0
            greyF = greyF.astype('uint8')
            greyF = np.take(greyF, mapping)

            flow = DISFlow.calc(prevF, greyF, flow)

            if keyboard.is_pressed('space'):
                homeMode = True

            if keyboard.is_pressed('w'):
                conX = self.fVel
            elif keyboard.is_pressed('s'):
                conX = -self.fVel

            if keyboard.is_pressed('q'):
                conRot = -90
            elif keyboard.is_pressed('e'):
                conRot = 90

            if keyboard.is_pressed('a'):
                side = -self.fVel
            elif keyboard.is_pressed('d'):
                side = self.fVel

            if keyboard.is_pressed('shift'):
                conX = conX * 2

            if keyboard.is_pressed('alt'):
                cooldown = time.time()

            sC = 0
            fC = self.fVel
            moveX, moveY, moveZ = self.relPosToAbs(conX, side, 0)
            if homeMode:
                moveX = 2 * homeVector[0]/abs(homeVector[0]) if abs(homeVector[0]) > 0.2 else 0
                moveY = 2 * homeVector[1]/abs(homeVector[1]) if abs(homeVector[1]) > 0.2 else 0

            homeVector[0] += 0.05 * -moveX
            homeVector[1] += 0.05 * -moveY

            newFlow = abs(flow)
            newFlow[:,:,1] = newFlow[:,:,1]
            detection = 0.8 * detection + 0.2 * newFlow.mean(axis=(0, 2))
            #detection[:, 0] -= np.mean(detection[:, 0]) # Removes purely rotational OF, technically should split up in u and v directions
            #detection[:, 1] -= np.mean(detection[:, 1])
            #detection[:, 0] += abs(min(detection[:, 0]))
            #detection[:, 1] += abs(min(detection[:, 1]))
            detection -= np.mean(detection)
            detection += abs(min(detection))
            normDetect = detection # Manhatten norm to reduce overhead
            forStrength = sum(normDetect * cosVals)
            sideStrength = sum(normDetect * sinVals)
            direction = math.degrees(math.atan(sideStrength/forStrength))
            strength = math.sqrt(1024 * (forStrength ** 2 + sideStrength ** 2) / len(detection))
            strength = strength * (1 - abs(direction)/180) ** 4 # This helps to prioritise head on threats!
            #print(round(strength), round(direction), round(max((1 - abs(direction)/90), 0.1) * 100)/ 100)
            _, _, curYaw = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
            curYaw = math.degrees(curYaw)

            #if (conRot != )

            #print(round(direction), "    ", round(strength*1000)/1000)
            if self.state == ms.intSac and time.time() - cooldown > self.cdLength and strength > self.turnTrig:
                self.state = ms.sac
                weight = 1 / (1 + (strength/self.threshold) ** (-self.gain))
                #print(direction)
                #absAvoid = self._clampRot(curYaw + direction)
                #angleDif = (((curYaw - absAvoid) + 180) % 360) - 180
                direction = self._clampRot(direction + 180)
                self.targetYaw = self._clampRot(curYaw + weight * direction)
                print(self.targetYaw, "    ", round(curYaw), "    ", round(strength * 1000) / 1000, "    ", weight)
                #print(self.targetYaw, curYaw)
                self.turnRate = 250 * direction/abs(direction)
            elif self.state == ms.sac and abs(curYaw - self.targetYaw) >= 5:
                conRot = self.turnRate
                #print(curYaw)
            elif self.state == ms.sac and abs(curYaw - self.targetYaw) < 5:
                self.state = ms.intSac
                print("DONE")
                cooldown = time.time()
                detection = np.zeros((greyF.shape[1])) # reset this bad boy
                conRot = 0

            if homeMode:
                client.moveByVelocityZAsync(moveX, moveY, -2, 0.05, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0))
            else:
                client.moveByVelocityZAsync(moveX, moveY, -2, 0.05, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, conRot))  # (targetYaw - curYaw) * 45) # airsim.DrivetrainType.MaxDegreeOfFreedom
            cv2.imshow('frame123', cv2.resize(greyF, (0, 0), fx=2, fy=2))
            cv2.imshow('frame124', cv2.resize(np.expand_dims(normDetect, 0), (0,0), fx=2, fy=64))
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('x'):
                break
            t1 = time.time()
            #print(round(1/(t1 - t0)))

drone = BioDrone()
drone.main()