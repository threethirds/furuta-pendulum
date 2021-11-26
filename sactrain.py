import gym
import time

from gym.wrappers import FrameStack

from stable_baselines3 import SAC

from real import FurutaPendulumEnv

env = FurutaPendulumEnv()

env.variable_init(steps = 1, coefficient = 0.4, timestep = 240)

env = FrameStack(env, 13)

file_name = "GGGPolicies/10000furutapendulum"

model = SAC.load(file_name)

model.set_env(env)

#model = SAC('MlpPolicy', env, verbose=1, gamma = 0.95, gradient_steps = 100, train_freq = 1500)

model.learn(total_timesteps=8000)

model.save("SACPolicies/00004furutapendulum.zip")

env.end()


