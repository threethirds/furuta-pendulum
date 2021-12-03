import gym
import time
from stable_baselines3 import PPO
from gym.wrappers import FrameStack

from pendulum.jetson import JetsonPendulum
from real import FurutaPendulumEnv
pendulum = JetsonPendulum(torque_coefficient=0.4)
env = FurutaPendulumEnv(pendulum, steps=1, timestep=120)
env = FrameStack(env, 13)

model = PPO('MlpPolicy', env, verbose=1) #,tensorboard_log="./sac_fpendulum_tensorboard/")

model.learn(total_timesteps=20000)

model.save("Policies/PPOfurutapendulum")
