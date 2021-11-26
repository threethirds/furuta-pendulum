import gym
import numpy as np

from stable_baselines3 import SAC

env = gym.make("LunarLander-v2")

model = SAC("MlpPolicy", env, verbose=1, gradient_steps = 50, train_freq = 1000)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_cartpole")


