import gym
import os
from os import listdir
import time
from datetime import datetime

from gym.wrappers import FrameStack
from stable_baselines3 import SAC

#Datetime for the creation of a folder (information is in the info folder)

def get_information():
    text_file = 'runs/info.txt'
    with open(text_file) as f:
        lines = f.readlines()
    
    current_datetime = int(lines[1].rstrip("\n"))
    example_datetime = lines[3].rstrip("\n")
    async_bool = int(lines[5].rstrip("\n"))
    additional_episodes = int(lines[7].rstrip("\n"))
    train_frequency = int(lines[9].rstrip("\n"))
    timeout = int(lines[11])
    f.close()

    if current_datetime:
        now = datetime.now()
        dt = now.strftime("%m_%d_%H_%M")
    else:
        dt = example_datetime
    return dt, additional_episodes, train_frequency, async_bool, timeout

#Find the largest file in the folder
def get_current_episode():
    text_file = 'runs/info.txt'
    with open(text_file) as f:
        lines = f.readlines()
    
    current_datetime = int(lines[1].rstrip("\n"))
    example_datetime = lines[3].rstrip("\n")
    f.close()
    
    if current_datetime:
        current_episode = 0
    else:
        if os.path.isdir("runs/"+example_datetime):
            list_of_files = listdir("runs/"+example_datetime+"/weights")
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

def setup_sac_learn(environment, current_episode, dt, total_timesteps, async_bool):
    policy_file = "runs/"+dt+"/weights/{}.zip".format(current_episode-async_bool)

    model = SAC.load(policy_file)
    model.set_env(environment)
    total_timesteps, callback = model._setup_learn(total_timesteps = total_timesteps, eval_env = environment, log_path = "log")
    return model, callback

def delete_file_path_pkl(dt, current_episode, recent_left, history):
    deleted_file_path = "runs/"+dt+"/data/{}.pkl".format(current_episode-recent_left)
    if os.path.isfile(deleted_file_path) and current_episode%history>0:
        os.unlink(deleted_file_path)

def delete_file_path_zip(dt, current_episode, recent_left, history):
    deleted_file_path = "runs/"+dt+"/weights/{}.zip".format(current_episode-recent_left)
    if os.path.isfile(deleted_file_path) and current_episode%history>0:
        os.unlink(deleted_file_path)