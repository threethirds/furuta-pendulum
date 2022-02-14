import os
from os import listdir

import time
from datetime import datetime

from rollout_backprop_definitions import get_information, get_current_episode, setup_sac_learn, delete_file_path_pkl
from gym.wrappers import FrameStack
from stable_baselines3 import SAC

from pendulum.jetson import JetsonPendulum
from real import FurutaPendulumEnv
from pendulum.mock import MockPendulum


#get_datetime gives us the name of the folder which is going to be used

dt, additional_episodes, train_frequency, async_bool, timeout = get_information()


#Creating the RL Environment
pendulum = JetsonPendulum(torque_coefficient=0.6)
#pendulum = MockPendulum()
env = FurutaPendulumEnv(pendulum, steps=1, timestep=120)
env = FrameStack(env, 13)

current_episode = get_current_episode()

total_episodes = current_episode + additional_episodes
total_timesteps = total_episodes*train_frequency


if current_episode==1:
    #set up the SAC model
    model = SAC('MlpPolicy', env, verbose=0, gamma = 0.99, gradient_steps = 500, train_freq = 1000, buffer_size=500000, optimize_memory_usage=True)

    total_timesteps, callback = model._setup_learn(total_timesteps = total_timesteps, eval_env = env, log_path = "log")
    if async_bool == 1:
        policy_file = "runs/"+dt+"/weights/0"
        model.save(policy_file)
    policy_file = "runs/"+dt+"/weights/1"
    model.save(policy_file)
else:
    policy_file = "runs/"+dt+"/weights/{}".format(current_episode)
    model = SAC.load(policy_file)
    model.set_env(env)
    total_timesteps, callback = model._setup_learn(total_timesteps = total_timesteps, eval_env = env, log_path = "log")
    if async_bool == 1:
        policy_file = "runs/"+dt+"/weights/{}".format(current_episode-1)
        model.save(policy_file)

log_interval = 4

timeconstraint = time.time() + timeout   

#timepoint
time_1  = time.time()


#async duplicates the first rollout (since we have two rollouts with weight0)

duplicate_first_rollout = 0
if async_bool == 1:
    duplicate_first_rollout = 1

#ROLLOUT loop function

while current_episode < total_episodes:

    policy_file = "runs/"+dt+"/weights/{}.zip".format(current_episode-async_bool)

    time.sleep(0.1)

    if os.path.isfile(policy_file):

        #neatsimenu, kodel reikalingas sis time.sleep

        time.sleep(0.2)

        model, callback = setup_sac_learn(environment = env, current_episode=current_episode-async_bool, dt=dt, total_timesteps = total_timesteps, async_bool=async_bool)

        #timepoint

        rollout = model.collect_rollouts(
                        model.env,
                        train_freq=model.train_freq,
                        action_noise=model.action_noise,
                        callback=callback,
                        learning_starts=model.learning_starts,
                        replay_buffer=model.replay_buffer,
                        log_interval=log_interval,
                    )
        
        #timepoint
        time_2  = time.time()
        print(time_2-time_1)
        time_1 = time_2

        data_file = "runs/"+dt+"/data/{}".format(current_episode)

        model.save_replay_buffer(data_file)

        deleted_file = delete_file_path_pkl(dt = dt, current_episode = current_episode, recent_left = 3, history = 666)

        #timepoint

        current_episode += 1

        if duplicate_first_rollout == 1: 
            current_episode -=1
            duplicate_first_rollout = 0


    if time.time() > timeconstraint:
        break