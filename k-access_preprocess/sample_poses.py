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

def get_gravity_vec(q):
    rot_ = Rotation.from_quat(q)
    mat_ = rot_.as_matrix()
    vec_ = np.dot(np.transpose(mat_), np.array([[0.], [0.], [-1.]]))
    return vec_

def reset_sim(friction = 0.7):
    p.resetSimulation()
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                 cameraTargetPosition=[-0.02, -0.09, 0.2])
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1. / 1000)
    planeId = p.loadURDF("plane.urdf",useFixedBase=True)
    p.changeDynamics(bodyUniqueId=planeId,linkIndex=-1,lateralFriction=friction*2,restitution=0.0)
    return planeId

def set_pos12(uid,arr): # positive = forward
    states = p.getJointStates(uid, joint_indice)
    cur_pos = np.array([s[0] for s in states])
    cur_velo = np.array([s[1] for s in states])
    _forces = 55 * (np.array(arr) - cur_pos) + 0.8 * (0 - cur_velo)
    _forces = np.clip(_forces, -maxForce, maxForce)
    p.setJointMotorControlArray(uid, joint_indice, forces=_forces, controlMode=p.TORQUE_CONTROL)

def get_state(uid):
    states = p.getJointStates(uid,joint_indice)
    pos = [s[0] for s in states]
    velo = [s[1] for s in states]
    react = [s[2] for s in states]
    torq = [s[3] for s in states]
    omega = p.getBaseVelocity(uid)[1]
    q = p.getBasePositionAndOrientation(uid)[1]
    grav = np.reshape(get_gravity_vec(q),(-1,))
    return pos,velo,list(omega),grav.tolist(),react,torq

physicsClient = p.connect(p.DIRECT)  #  p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
a1_path = r'../unitree_ros/robots/a1_description/urdf/a1.urdf'
np.random.seed(42)

num_samples = 0
save_poses = []
while num_samples < 1000:
    planeId = reset_sim()
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.setGravity(0, 0, -9.81)
    StartPos = [0, 0, a1height+0.5]
    q = np.random.uniform([-pi,-pi/2,0],[pi,pi/2,0])
    StartOrientation = p.getQuaternionFromEuler(q)
    # StartOrientation = p.getQuaternionFromEuler(rand_q)
    a1id = p.loadURDF(a1_path, StartPos, StartOrientation,useFixedBase=False,
                          flags=p.URDF_USE_SELF_COLLISION)

    self_collision = True
    while self_collision:
        self_collision = False
        print('resample')
        rst_pos = np.random.uniform(lowers, uppers)
        for i in range(12):
            p.resetJointState(a1id,joint_indice[i],rst_pos[i])
        cp_list = p.getClosestPoints(a1id,a1id,0.005)
        for cp in cp_list:
            if (cp[3],cp[4]) not in collision_pairs_born:
                self_collision = True
                break

    for i in range(12):
        p.setJointMotorControl2(a1id, joint_indice[i], p.VELOCITY_CONTROL, force=0)


    for i in range(1000*4):
        p.stepSimulation()
        set_pos12(a1id,rst_pos)

    to_gnd_dist = np.min([x[8] for x in p.getClosestPoints(a1id,planeId,100)])
    if np.abs(to_gnd_dist) < 1e-3:
        store_height = p.getBasePositionAndOrientation(a1id)[0][2] + 1e-3
        store_roll_pitch = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(a1id)[1])[:2]
        store_joints = [s[0] for s in p.getJointStates(a1id, joint_indice)]
        store_dict = {'store_height':store_height, 'store_roll_pitch':store_roll_pitch, 'store_joints':store_joints}
        save_poses.append(store_dict)
        num_samples += 1
        print('# sample: %d'%num_samples)

with open('poses.pickle', 'wb') as handle:
    pickle.dump(save_poses, handle, protocol=pickle.HIGHEST_PROTOCOL)

p.disconnect()
