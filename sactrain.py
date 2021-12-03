from gym.wrappers import FrameStack
from stable_baselines3 import SAC

from pendulum.jetson import JetsonPendulum
from real import FurutaPendulumEnv

file_name = "Policies/10003furutapendulum15h_1h"

pendulum = JetsonPendulum(torque_coefficient=0.4)
env = FurutaPendulumEnv(pendulum, steps=1, timestep=120)
env = FrameStack(env, 13)

# model = SAC('MlpPolicy', env, verbose=1, gamma = 0.95, gradient_steps = 100, train_freq = 1500)

model = SAC.load(file_name)
model.set_env(env)

model.learn(total_timesteps=20000)
model.save("Policies/R00001furutapendulum_sim15h1h.zip")
env.end()
