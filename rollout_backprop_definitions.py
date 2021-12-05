import gym
import os
from os import listdir
import time
from datetime import datetime

from gym.wrappers import FrameStack
from stable_baselines3 import SAC

#Datetime for the creation of a folder

def get_datetime(current_datetime = True, example_datetime = "12_04_21_54"):
    if current_datetime:
        now = datetime.now()
        dt = now.strftime("%m_%d_%H_%M")
    else:
        dt = example_datetime
    return dt, example_datetime

#Find the largest file in the folder
def get_current_episode(start_from_scratch = True, directory = "waights/", example_folder = "12_04_21_54"):
    if start_from_scratch:
        current_episode = 0
    else:
        if os.path.isdir(directory+example_folder):
            list_of_files = listdir(directory+example_folder)
            length = len(list_of_files)
            for i in range(length):
                x = list_of_files[i]
                s = x[:-4]
                list_of_files[i] = int(s)
            file_number = max(list_of_files)
            current_episode = file_number
            print(current_episode)
        else:
            current_episode = 0
    return current_episode

def setup_rollout_sac(environment, dt, additional_episodes = 10, train_frequency = 1500, gradient_steps = 100, buffer_size = 100000, memory_optimization = True, current_episode = 0):
    total_episodes = current_episode + additional_episodes
    total_timesteps = total_episodes*train_frequency

    if current_episode==0:
        #set up the SAC model
        model = SAC('MlpPolicy', env, verbose=1, gamma = 0.99, gradient_steps = 100, train_freq = train_frequency, buffer_size=buffer_size, optimize_memory_usage=memory_optimization)

        total_timesteps, callback = model._setup_learn(total_timesteps = total_timesteps, eval_env = environment, log_path = "log")
        policy_file = "waights/"+dt+"/0"
        model.save(policy_file)
    else:
        policy_file = "waights/"+dt+"/{}".format(current_episode)
        model = SAC.load(policy_file)
        model.set_env(environment)
        total_timesteps, callback = model._setup_learn(total_timesteps = total_timesteps, eval_env = environment, log_path = "log")

    log_interval = 4

    return total_episodes, model, total_timesteps, callback, log_interval

def setup_sac_learn(environment, current_episode, dt, total_timesteps):
    policy_file = "waights/"+dt+"/{}.zip".format(current_episode)

    model = SAC.load(policy_file)
    model.set_env(environment)
    total_timesteps, callback = model._setup_learn(total_timesteps = total_timesteps, eval_env = environment, log_path = "log")
    return model, callback

def delete_file_path_pkl(directory, dt, current_episode, recent_left, history):
    deleted_file_path = directory+dt+"/{}.pkl".format(current_episode-recent_left)
    if os.path.isfile(deleted_file_path) and current_episode%history>0:
        os.unlink(deleted_file_path)

def delete_file_path_zip(directory, dt, current_episode, recent_left, history):
    deleted_file_path = directory+dt+"/{}.zip".format(current_episode-recent_left)
    if os.path.isfile(deleted_file_path) and current_episode%history>0:
        os.unlink(deleted_file_path)