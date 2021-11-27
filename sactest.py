from gym.wrappers import FrameStack
from stable_baselines3 import SAC

from pendulum.jetson import JetsonPendulum
from real import FurutaPendulumEnv

file_name = "GGGPolicies/2021-09-07 14_24_00_120.0tss_4steps_0.3torque_12frames_0.99gamma 00001"

test_steps = 2000

t = 0
i = 0

timestep = int(file_name[32:35 + t])
steps = int(file_name[41 + t])
frames = int(file_name[58 + i + t:60 + i + t])
iteration = file_name[77 + i + t:82 + i + t]

pendulum = JetsonPendulum(torque_coefficient=1)
env = FurutaPendulumEnv(pendulum, steps=steps, timestep=timestep)
env = FrameStack(env, frames)

model = SAC('MlpPolicy', env, verbose=1, gamma=0.99)
model = model.load(file_name)

obs = env.reset()
for _ in range(test_steps):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

env.end()

# TODO properly log & stream data
