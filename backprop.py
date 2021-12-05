import gym
import os
from os import listdir

import time
from datetime import datetime

import numpy as np
import torch as th

from torch.nn import functional as F
from gym.wrappers import FrameStack

from stable_baselines3.common.utils import polyak_update
from stable_baselines3 import SAC

from rollout_backprop_definitions import get_datetime, get_current_episode, setup_rollout_sac, setup_sac_learn, delete_file_path_zip


#Creating the RL Environment
from pendulum.mock import MockPendulum
from real import FurutaPendulumEnv

pendulum = MockPendulum()
env = FurutaPendulumEnv(pendulum, steps=1, timestep=120)
env = FrameStack(env, 13)

#Datetime for the creation of a folder

dt, example_folder = get_datetime(current_datetime = True, example_datetime = "12_05_19_54")

#Find the largest file in the folder
current_episode = get_current_episode(directory = "deta/", example_folder = example_folder, start_from_scratch = True)

#TODO implement normal total episodes
additional_episodes = 10
total_episodes = current_episode+additional_episodes
train_frequency = 1500
total_timesteps = total_episodes*train_frequency

#TODO Implement normal time break
timeout = time.time() + 10*60   # 10 minutes from now
print(current_episode)
print(dt)
while current_episode < total_episodes:
    data_file = "deta/"+dt+"/{}.pkl".format(current_episode)

    if os.path.isfile(data_file):

        policy2_file = "waights/"+dt+"/{}".format(current_episode+1)

        #set up the SAC model

        model, callback = setup_sac_learn(environment = env, current_episode=current_episode, dt=dt, total_timesteps= total_timesteps)

        model.load_replay_buffer(data_file)

        model.train(gradient_steps = 100)

        model.save(policy2_file)

        deleted_file = delete_file_path_zip(directory = "waights/", dt = dt, current_episode = current_episode, recent_left = 1, history = 10)

        current_episode += 1

    if time.time() > timeout:
        break