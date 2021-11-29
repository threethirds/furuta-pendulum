from gym.wrappers import FrameStack
from stable_baselines3 import SAC

from pendulum.mock import MockPendulum
from real import FurutaPendulumEnv

pendulum = MockPendulum()
env = FurutaPendulumEnv(pendulum, steps=1, timestep=240)
env = FrameStack(env, 13)

model = SAC('MlpPolicy', env, verbose=1, gamma=0.95, gradient_steps=100, train_freq=1500)
model.set_env(env)

model.learn(total_timesteps=8000)
env.end()
