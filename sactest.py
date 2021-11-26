import gym
import time

import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from gym.wrappers import FrameStack
from datetime import datetime

from real import FurutaPendulumEnv
env = FurutaPendulumEnv()

file_name = "GGGPolicies/2021-09-07 14_24_00_120.0tss_4steps_0.3torque_12frames_0.99gamma 00001"

test_steps = 2000

t = 0
i = 0

path_in_str = file_name

timestepsize = path_in_str[32:35+t]
stepssize = path_in_str[41+t]
frames = path_in_str[58+i+t:60+i+t] #2digits
iteration = path_in_str[77+i+t:82+i+t]
#print(timestepsize, stepssize, frames)

env.variable_init(steps = int(stepssize), coefficient = 1, timestep = int(timestepsize))

env = FrameStack(env, int(frames))

model = SAC('MlpPolicy', env, verbose=1, gamma=0.99)

model = model.load(file_name)

obs = env.reset()
time1 = []
rewardsa = []
time2 = []
for _ in range(test_steps):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    #actions.append(info["action"])
    #sines.append(info["sine"])
    #cosines.append(info["cosine"])
    rewardsa.append(info["reward"])

    #print(info["alpha"])

    #times.append(info["step time"])
    #real_actions.append(info["actual action"])
env.end()
t = list(range(0,test_steps))
'''
fig, ax = plt.subplots(2, 2)

ax[0,0].plot(t, times, 'k')
ax[0,0].set_title('Timestep')
ax[1,0].plot(t, sines, color='r', label='sin')
ax[1,0].plot(t, cosines, color='g', label='cos')
ax[0,1].plot(t, rewardsa)

ax[0,1].set_title('Reward')
ax[1,1].plot(t, actions)
ax[1,1].plot(t, real_actions, color = 'm')
'''

plt.plot(t, rewardsa)

now = datetime.now()

dt = now.strftime("%Y-%m-%d_%H:%M:%S")

plt.savefig("Plots/reward"+iteration+dt+".png", dpi=200)
#plt.savefig("Plots/reward"+dt+".png", dpi=200)
'''
plt.clf()

time1, = plt.plot(t, time1)
time2, = plt.plot(t, time2)

plt.legend([time1, time2], ['full step time', 'predict time'])

plt.savefig("Plots/times"+iteration+dt+".png", dpi=200)
'''


