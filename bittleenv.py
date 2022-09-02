import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import pybullet as p
import pybullet_data
import math
from math import pi
import numpy as np
import random
from scipy.spatial.transform import Rotation
import time
import pandas as pd
from poseconfigs import *
joint_indice = [x[0] for x in specs]
specs = np.array(specs)
maxForce, maxVelo = specs[0,3], specs[0,4]
print(specs)
dyn_link_masses = masses

def get_omega_imu(q,omega):
    rot_ = Rotation.from_quat(q)
    mat_ = rot_.as_matrix()
    vec_ = np.dot(np.transpose(mat_), np.array([[x] for x in omega]))
    return np.reshape(vec_,(-1,))

def get_gravity_vec(q):
    rot_ = Rotation.from_quat(q)
    mat_ = rot_.as_matrix()
    vec_ = np.dot(np.transpose(mat_), np.array([[0.], [0.], [-1.]]))
    out_ = np.reshape(vec_, (-1,))
    return out_

def rbf_reward(x,xhat,alpha):
    x = np.array(x)
    xhat = np.array(xhat)
    return np.exp(alpha * np.sum(np.square(x-xhat)))


class Bittle(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,connect_GUI=False):
        self.gui = connect_GUI
        self.bittle_path = r'/home/zc/terrain/fallrec/unitree_ros/robots/a1_description/urdf/a1.urdf'
        self.mp4log=0
        if connect_GUI:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=20, cameraPitch=-10,
                                     cameraTargetPosition=[0.05, -0.15, 0.05])
        self.action_space = spaces.Box(specs[:,1], specs[:,2])
        self.observation_space = spaces.Box(np.array([-1]*3 + specs[:,1].tolist() + [-15]*3 ),
                                            np.array([ 1]*3 + specs[:,2].tolist() + [ 15]*3 )  )
        self.cnt_ = 0
        self.cnt2 = 0

        self.configs = kaccess_config + ninepose_config  # rnd_config
        print('# initial poses:',len(self.configs))

        self.test_pos = ([0,0,0.057], [0, 1.0, -1.8]*4, [pi,0,0])

        self.maxForce = maxForce
        self.maxVelo = maxVelo
        self.friction = 0.7
        self.dr = False  # false, or 0 to 1

    def step(self,action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        old_pos = self.get_state()[0]
        new_pos = np.array(action)
        new_pos = np.clip(new_pos, specs[:,1], specs[:,2])
        real_pos = new_pos
        self.torq = np.zeros((12,))
        for s in range(20):
            self.set_pos12(real_pos)
            p.stepSimulation() # 20/500 = 1/25
            if self.gui:
                pass
        self.torq/= 20

        pos, velo, omega, grav, react, torq = self.get_state()
        observation = grav + pos + omega

        base_velo = p.getBaseVelocity(self.bittleId)[0]
        height = p.getBasePositionAndOrientation(self.bittleId)[0][2]
        # q = p.getBasePositionAndOrientation(self.bittleId)[1]
        # roll_ = p.getEulerFromQuaternion(q)[0]

        cos_ori = grav[2] * (-1)
        ori_thres = (cos_ori > np.cos(0.25*pi))
        ori_reward = 0.667 * (0.5 * cos_ori + 0.5) ** 2 # used

        height_reward = 0.333 * np.clip( (height-0.06)/0.25, 0.0, 1.0) * ori_thres
        pose_weights = np.array([2.0, 1.0, 0.8] * 4)
        tgt_pose = np.array([0, 0.7, -1.5] * 4) * pose_weights
        pose_weighted = np.array(pos) * pose_weights
        nominal_pose_reward = 0.667 * (1-np.sum(np.square(pose_weighted-tgt_pose))/20) * ori_thres

        standing_velo_regu = 0.067 * rbf_reward(velo, 0, -0.02)
        stand_reward = height_reward + nominal_pose_reward + standing_velo_regu # used

        foot_contact, body_contact = self.contact_reward()
        foot_contact_reward = 0.067 * foot_contact  # used
        body_contact_reward = 0.067 * body_contact  # used

        dists = [c[8] for c in p.getClosestPoints(self.bittleId, self.planeId, 1.)]
        # use 6,11,16,21
        try:
            foot_height_reward = 0.333 * rbf_reward(dists[6:22:5],[0]*4,-200.0) * ori_thres  # used
        except:
            foot_height_reward = -1.

        torq = self.torq
        joint_torq_regu_reward = 0.033 * rbf_reward(torq,0,-0.002)  # used
        delta_pose_reward = 0.067 * rbf_reward(new_pos,old_pos,-0.5)  * 0   # used
        
        no_flip = - 10 * ( np.abs(grav[0])>np.cos(0.2*pi) )  # used
        med_pos, range_pos = (specs[:,1] + specs[:,2])/2, (specs[:,2] - specs[:,1])/2
        pos_bias = (np.array(pos) - med_pos) / range_pos
        boundary_regu = - 100 * np.sum((pos_bias>0.95) * np.square(pos_bias-0.95)) # used

        cp_list = p.getClosestPoints(self.bittleId,self.bittleId,0.005)
        cp_num = 0
        for cp in cp_list:
            if (cp[3],cp[4]) not in collision_pairs_born:
                cp_num += 1
        selfCol_regu = -0.05 * cp_num  # used

        reward = ori_reward + stand_reward + foot_contact_reward + body_contact_reward + foot_height_reward \
                 + joint_torq_regu_reward + delta_pose_reward + no_flip + boundary_regu + selfCol_regu
        reward = reward * 0.04   # dt
        info = {'ori_reward':ori_reward, 'ori_thres':ori_thres, 'selfCol_regu': selfCol_regu,
                'height_reward':height_reward, 'nominal_pose_reward':nominal_pose_reward,
                'standing_velo_regu':standing_velo_regu, 'stand_reward':stand_reward,
                'foot_contact_reward':foot_contact_reward, 'body_contact_reward':body_contact_reward,
                'joint_torq_regu_reward':joint_torq_regu_reward, 'boundary_regu':boundary_regu,
                'delta_pose_reward':delta_pose_reward, 'no_flip':no_flip, 'foot_height_reward':foot_height_reward,
                }
        if self.gui:
            # print(info['foot_contact'],info['body_contact'],info['height_reward'],
            #       info['no_pitch'],info['nominal_pose_reward'])
            pass
        if reward > 9999:
            done = True # never done
        else:
            done = False

        return observation, reward, done, info

    def reset(self, evaluate = False):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.81)
        # p.setPhysicsEngineParameter(fixedTimeStep= 1./500.)
        p.setTimeStep(1. / 500.)

        self.planeId = p.loadURDF("plane.urdf", useFixedBase=True)
        if self.gui:
            pass
            #self.mp4log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,'record'+'.mp4')

        # config_num  = random.randint(0,len(self.configs)-1)
        config_num = self.cnt_ % (len(self.configs))
        if evaluate == False:
            self.cnt_ = self.cnt_ + 1
        self.StartPos, rst_pos,rpy_ini = self.configs[config_num]

        if evaluate == True:
            config_num  = self.cnt2 % 1
            self.cnt2 += 1
            self.StartPos, rst_pos, rpy_ini = self.test_pos

        print('initialize with', config_num, 'domain randomization:',self.dr)
        self.StartOrientation = p.getQuaternionFromEuler(rpy_ini)
        self.bittleId = p.loadURDF(self.bittle_path, self.StartPos, self.StartOrientation,
                                   useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE)

        ########### domain randomization
        if self.dr:
            self.friction = np.random.uniform(0.7 - self.dr * 0.4, 0.7 + self.dr * 0.8)  # 0.3 to 1.5
            self.maxForce = maxForce * np.random.uniform(1.0 - self.dr * 0.2, 1.0 + self.dr * 0.2)  # 0.8 to 1.2
            self.maxVelo = maxVelo * np.random.uniform(1.0 - self.dr * 0.3, 1.0 + self.dr * 0.2)   # 0.7 to 1.2
            for link_id,link_mass in dyn_link_masses.items():
                p.changeDynamics(self.bittleId, link_id,
                                 mass = link_mass * np.random.uniform(1.0 - self.dr * 0.2, 1.0 + self.dr * 0.2) ) # 0.8 to 1.2
        p.changeDynamics(bodyUniqueId=self.planeId, linkIndex=-1, lateralFriction= 2.0 * self.friction, restitution=1.0)

        for i in range(12):
            p.resetJointState(self.bittleId,joint_indice[i],rst_pos[i])
        for i in range(12):
            p.setJointMotorControl2(self.bittleId, joint_indice[i], p.VELOCITY_CONTROL, force=0)

        pos, velo, omega, grav, react, torq = self.get_state()
        observation = grav + pos + omega
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return observation

    def render(self, mode='human'):
        pass
        return

    def close(self):
        if self.gui:
            #p.stopStateLogging(self.mp4log)
            pass
        p.disconnect()

    def contact_reward(self):
        """return foot,body
        foot = 1 if in contact with ground, otherwise 0
        body = 0 if in contact with ground, otherwise 1
        """
        clp = p.getClosestPoints(self.bittleId, self.planeId, 5e-3)
        links = [c[3] for c in clp]
        
        foot = 0 # foot - ground
        if (6 in links):
            foot += 1.0
        if (11 in links):
            foot += 1.0
        if (16 in links):
            foot += 1.0
        if (21 in links):
            foot += 1.0

        if (0 in links):
            body = 0  # body - ground
        else:
            body = 1
        return foot,body
    def set_pos12(self, arr):  # positive = forward
        states = p.getJointStates(self.bittleId, joint_indice)
        cur_pos = np.array([s[0] for s in states])
        cur_velo = np.array([s[1] for s in states])
        _forces = 55 * (np.array(arr)-cur_pos) + 0.8 * (0-cur_velo)   #  80/1.5 vs 55/0.8
        _forces = np.clip(_forces,-self.maxForce, self.maxForce)
        p.setJointMotorControlArray(self.bittleId, joint_indice, forces = _forces,
                                controlMode=p.TORQUE_CONTROL)
        self.torq += _forces

    def get_state(self):
        states = p.getJointStates(self.bittleId, joint_indice)
        pos = [s[0] for s in states]
        velo = [s[1] for s in states]
        react = [s[2] for s in states]
        torq = [s[3] for s in states]  # is zero for pd control
        omega = list(p.getBaseVelocity(self.bittleId)[1])  # base omega
        q = p.getBasePositionAndOrientation(self.bittleId)[1]
        grav = get_gravity_vec(q).tolist()
        omega = get_omega_imu(q,omega).tolist()  # imu omega
        
        return pos, velo, omega, grav, react, torq
