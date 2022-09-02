import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from bittleenv import Bittle
import time


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="bittle_fall_recovery")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.987, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.001)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=3000001, metavar='N',
                    help='maximum number of steps (default: 3000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", default=torch.cuda.is_available(),
                    help='run on CUDA (default: is_available)')
args = parser.parse_args()
args.cuda = False

# Environment
env = Bittle(connect_GUI=True)
# env = Bittle(connect_GUI=False)
env._max_episode_steps = 500*2+10 # for sample/estimation # 75 # rollout
env.dr = False
env.friction = 0.7
# env.test_pos = ([0,0,0.5], [0, 2.0, -1.0]*4, [-2.5,0.8,0])
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
model_name = 'rndpose_' * 0 +'seed123456'   #  _____________________________________  edit __________
# model_name = 'bittle_rndpose4feet1600'
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model(env_name=model_name)

episodes = 1
rewards = []

for _ in range(episodes):
    state = env.reset(evaluate=True)
    prev_action = None

    episode_reward = 0
    done = False
    episode_steps = 0
    reward = 0
    prev_action = None
    time.sleep(1)

    st_time = time.monotonic()
    while (episode_steps < env._max_episode_steps):
        action = agent.select_action(state, evaluate=True)

        thr_ = 0.06
        if reward > thr_:
            ratio = np.clip((reward-thr_)/(0.09-thr_),0,1)
            action = ratio * np.array([0, 0.7, -1.5] * 4) + (1-ratio) * action

        action = np.array(action)
        if prev_action is None:
            prev_action = action + 0.
        # print(action[1::3])
        action = 0.6 * action + 0.4 * prev_action
        next_state, reward, done, info = env.step(action)
        prev_action = action

        print(episode_steps*0.04, reward, info['joint_torq_regu_reward'],info['nominal_pose_reward'],info['foot_height_reward'])
        episode_reward += reward
        state = next_state
        episode_steps += 1
        while time.monotonic() - st_time < episode_steps*0.04:
            time.sleep(0.0001)

    print('episode reward:',round(episode_reward,2),'last step reward:',round(reward,4))
    rewards.append(episode_reward)

env.close()
