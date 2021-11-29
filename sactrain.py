import gym
import time

from gym.wrappers import FrameStack

from stable_baselines3 import SAC

from real import FurutaPendulumEnv

env = FurutaPendulumEnv()

env.variable_init(steps = 1, coefficient = 0.4, timestep = 240)

env = FrameStack(env, 13)

file_name = "Policies/00004furutapendulum"

model = SAC.load(file_name)

model.set_env(env)

#model = SAC('MlpPolicy', env, verbose=1, gamma = 0.95, gradient_steps = 100, train_freq = 1500)

model.learn(total_timesteps=8000)

model.save("Policies/00005furutapendulum.zip")

env.end()


