"""
Author: Chong Zhang
EMail: chozhang@ethz.ch
Date: 29/8/2022
"""

import pybullet as p
import time
import pybullet_data
import math
from math import pi
import numpy as np
from scipy.spatial.transform import Rotation
import pickle
import sys
sys.path.insert(0,'..')
from poseconfigs import *

joint_indice = [x[0] for x in specs]
specs = np.array(specs)
maxForce = specs[0][3]
lowers = specs[:,1]
uppers = specs[:,2]
a1height=0.48
a1Path = r'../unitree_ros/robots/a1_description/urdf/a1.urdf'
with open('poses.pickle', 'rb') as handle: poses = pickle.load(handle)
print('# poses: %d' % len(poses))

from multiprocessing import Pool

def do(batchInput):
    physicsClient = p.connect(p.DIRECT)  #  p.connect(p.GUI)    # physicsClientId = physicsClient
    print("Client:",physicsClient)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

    def get_gravity_vec(q):
        rot_ = Rotation.from_quat(q)
        mat_ = rot_.as_matrix()
        vec_ = np.dot(np.transpose(mat_), np.array([[0.], [0.], [-1.]]))
        return vec_

    def reset_sim(friction = 0.7):
        p.resetSimulation(physicsClientId = physicsClient)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
        #                              cameraTargetPosition=[-0.02, -0.09, 0.2],physicsClientId = physicsClient)
        p.setGravity(0, 0, -9.81,physicsClientId = physicsClient)
        p.setTimeStep(1. / 500.,physicsClientId = physicsClient)
        planeId = p.loadURDF("plane.urdf",useFixedBase=True,physicsClientId = physicsClient)
        p.changeDynamics(bodyUniqueId=planeId,linkIndex=-1,lateralFriction=friction * 2, restitution=0.7,physicsClientId = physicsClient)
        return planeId

    def set_pos12(uid, arr, tgt_velo = 0): # positive = forward
        states = p.getJointStates(uid, joint_indice,physicsClientId = physicsClient)
        cur_pos = np.array([s[0] for s in states])
        cur_velo = np.array([s[1] for s in states])
        _forces = 55 * (np.array(arr) - cur_pos) + 0.8 * (tgt_velo - cur_velo)
        _forces = np.clip(_forces, -maxForce, maxForce)
        p.setJointMotorControlArray(uid, joint_indice, forces=_forces, controlMode=p.TORQUE_CONTROL,
                                    physicsClientId = physicsClient)

    def get_state(uid):
        states = p.getJointStates(uid,joint_indice,physicsClientId = physicsClient)
        pos = [s[0] for s in states]
        velo = [s[1] for s in states]
        react = [s[2] for s in states]
        torq = [s[3] for s in states]
        omega = p.getBaseVelocity(uid,physicsClientId = physicsClient)[1]
        q = p.getBasePositionAndOrientation(uid,physicsClientId = physicsClient)[1]
        grav = np.reshape(get_gravity_vec(q),(-1,))
        return pos,velo,list(omega),grav.tolist(),react,torq

    def fetch_time(src: int, dst: int, poseLib: list = poses, jointIndice: list = joint_indice) -> float:
        """
        src: int, the index of the source pose
        dst: int, the index of the destination pose
        poseLib: list, contains the library of stored poses, wherein a dict contains keys 'store_height': height,
                'store_roll_pitch': (roll, pitch), 'store_joints': [joint0, ..., joint12]
        jointIndice: list, mapping the action of joints to link id
        """
        srcPose, dstPose = poseLib[src], poseLib[dst]
        startJoints, dstJoints = srcPose['store_joints'], dstPose['store_joints']
        startPos, oriQ = [0, 0, srcPose['store_height']], list(srcPose['store_roll_pitch'])
        oriQ.append(0)
        startOrientation = p.getQuaternionFromEuler(oriQ)
        dstQ = list(dstPose['store_roll_pitch'])
        dstQ.append(0)
        dstOri = p.getQuaternionFromEuler(dstQ)
        dstGrav = get_gravity_vec(dstOri)[:,0]

        planeId = reset_sim()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId = physicsClient)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0, physicsClientId = physicsClient)

        a1Id = p.loadURDF(a1Path, startPos, startOrientation, useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION,physicsClientId = physicsClient)
        for i in range(12):
            p.resetJointState(a1Id, jointIndice[i], startJoints[i],physicsClientId = physicsClient)
        for i in range(12):
            p.setJointMotorControl2(a1Id, jointIndice[i], p.VELOCITY_CONTROL, force=0, physicsClientId = physicsClient)
        set_pos12(a1Id, startJoints)
        p.stepSimulation(physicsClientId = physicsClient)

        dist = 20
        for _t in range(500*20):
            progress = np.clip(_t/1000, 0, 1)
            action = np.array(startJoints) * (1-progress) + np.array(dstJoints) * progress
            set_pos12(a1Id, action)
            p.stepSimulation(physicsClientId = physicsClient)

            states = p.getJointStates(a1Id, jointIndice,physicsClientId = physicsClient)
            curPos = np.array([s[0] for s in states])
            posDist = np.sum(np.square(np.array(dstJoints) - curPos)) ** 0.5
            curH, curQ = p.getBasePositionAndOrientation(a1Id, physicsClientId=physicsClient)[0][2], \
                         p.getBasePositionAndOrientation(a1Id, physicsClientId=physicsClient)[1]
            hDist = np.abs(dstPose['store_height'] - curH)
            curGrav = get_gravity_vec(curQ)[:, 0]
            gravDist = np.sum(np.square(dstGrav-curGrav)) ** 0.5
            # if _t % 100 ==0:
            #     print(posDist, hDist, gravDist)
            if posDist < 0.5 and hDist < 0.01 and gravDist < 0.1:
                dist = _t/500.
                break

        return dist

    stTime = time.time()
    # try:
    #     batch = eval(sys.argv[1])
    # except:
    #     batch = eval(input('\n\n\n the batch:'))
    batch = batchInput
    bs, len_col = 22, 300  # zc
    accessTimeTable = np.zeros((len_col,len_col))
    print('running the rows %d to %d-1'%(bs*batch, min(len_col, bs*(batch+1)) ))
    for i in range(bs*batch, min(len_col, bs*(batch+1)) ):
        for j in range(len_col):
            if j!=i:
                accessTime = fetch_time(i,j)
                print('from',i,'to',j,'access',accessTime,'time:%.2f'%(time.time()-stTime))
                accessTimeTable[i,j] = accessTime
    np.save('accessTimeTable'+str(batch), accessTimeTable)

    p.disconnect(physicsClientId=physicsClient)


if __name__ == '__main__':
    proPool = Pool(14)  # zc
    proPool.map(do,list(range(14)))  # zc
