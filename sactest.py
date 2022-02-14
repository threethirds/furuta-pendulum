from gym.wrappers import FrameStack
from stable_baselines3 import SAC

from pendulum.jetson import JetsonPendulum
from real import FurutaPendulumEnv

file_name = "Policies/00001furutapendulum"

test_steps = 2000


pendulum = JetsonPendulum(torque_coefficient=0.7)
env = FurutaPendulumEnv(pendulum, steps=1000, timestep=120)
env = FrameStack(env, 13)

model = SAC('MlpPolicy', env, verbose=1, gamma=0.99)
model = model.load(file_name)

obs = env.reset()
for _ in range(test_steps):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

env.end()

# TODO properly log & stream data
