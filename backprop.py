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

from rollout_backprop_definitions import get_information, get_current_episode, setup_sac_learn, delete_file_path_zip


#Creating the RL Environment
from pendulum.mock import MockPendulum
from real import FurutaPendulumEnv

pendulum = MockPendulum()
env = FurutaPendulumEnv(pendulum, steps=1, timestep=120)
env = FrameStack(env, 13)

#Datetime for the creation of a folder

dt, additional_episodes, train_frequency, async_bool, timeout = get_information()

#Find the largest file in the folder
current_episode = get_current_episode()

#TODO implement normal total episodes

total_episodes = current_episode+additional_episodes
total_timesteps = total_episodes*train_frequency

#TODO Implement normal time break
timerestriction = time.time() + timeout

while current_episode < total_episodes:
    data_file = "runs/"+dt+"/data/{}.pkl".format(current_episode)

    if os.path.isfile(data_file):

        policy2_file = "runs/"+dt+"/weights/{}".format(current_episode+1)

        #set up the SAC model

        model, callback = setup_sac_learn(environment = env, current_episode=current_episode, dt=dt, total_timesteps= total_timesteps, async_bool=async_bool)

        model.load_replay_buffer(data_file)

        model.train(gradient_steps = 500)

        model.save(policy2_file)

        deleted_file = delete_file_path_zip(dt = dt, current_episode = current_episode, recent_left = 2, history = 10)

        current_episode += 1

    if time.time() > timerestriction:
        break