import os
from os import listdir

import time
from datetime import datetime

from rollout_backprop_definitions import get_datetime, get_current_episode, setup_rollout_sac, setup_sac_learn, delete_file_path_pkl
from gym.wrappers import FrameStack
from stable_baselines3 import SAC

from pendulum.jetson import JetsonPendulum
from real import FurutaPendulumEnv
from pendulum.mock import MockPendulum

#Datetime for the creation of a folder

start_time  = time.time()

dt, example_folder = get_datetime(current_datetime = True, example_datetime = "12_05_19_54")


#Creating the RL Environment
#pendulum = JetsonPendulum(torque_coefficient=0.4)
pendulum = MockPendulum()
env = FurutaPendulumEnv(pendulum, steps=1, timestep=120)
env = FrameStack(env, 13)

current_episode = get_current_episode(directory = "waights/", example_folder = example_folder, start_from_scratch = True)

environment = env
additional_episodes = 10
train_frequency = 1500
buffer_size = 100000
memory_optimization = True

total_episodes = current_episode + additional_episodes
total_timesteps = total_episodes*train_frequency

time_2  = time.time()
print(time_2-start_time)

if current_episode==0:
    #set up the SAC model
    model = SAC('MlpPolicy', env, verbose=0, gamma = 0.99, gradient_steps = 100, train_freq = train_frequency, buffer_size=buffer_size, optimize_memory_usage=memory_optimization)

    total_timesteps, callback = model._setup_learn(total_timesteps = total_timesteps, eval_env = environment, log_path = "log")
    policy_file = "waights/"+dt+"/0"
    model.save(policy_file)
else:
    policy_file = "waights/"+dt+"/{}".format(current_episode)
    model = SAC.load(policy_file)
    model.set_env(environment)
    total_timesteps, callback = model._setup_learn(total_timesteps = total_timesteps, eval_env = environment, log_path = "log")

log_interval = 4

#TODO Implement normal time break
timeout = time.time() + 10*60   # 10 minutes from now

time_3  = time.time()
print(time_3-time_2)

while current_episode < total_episodes:

    policy_file = "waights/"+dt+"/{}.zip".format(current_episode)

    if os.path.isfile(policy_file):

        time.sleep(0.2)

        model, callback = setup_sac_learn(environment = env, current_episode=current_episode, dt=dt, total_timesteps = total_timesteps)

        time_4  = time.time()
        print(time_4-time_3)

        rollout = model.collect_rollouts(
                        model.env,
                        train_freq=model.train_freq,
                        action_noise=model.action_noise,
                        callback=callback,
                        learning_starts=model.learning_starts,
                        replay_buffer=model.replay_buffer,
                        log_interval=log_interval,
                    )
        
        time_5  = time.time()
        print(time_5-time_4)

        data_file = "deta/"+dt+"/{}".format(current_episode)

        model.save_replay_buffer(data_file)

        deleted_file = delete_file_path_pkl(directory = "deta/", dt = dt, current_episode = current_episode, recent_left = 2, history = 10)

        time_3  = time.time()
        print(time_3-time_5)

        

        current_episode += 1

    if time.time() > timeout:
        break