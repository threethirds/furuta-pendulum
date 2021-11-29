from gym.wrappers import FrameStack
from stable_baselines3 import SAC

from pendulum.jetson import JetsonPendulum
from real import FurutaPendulumEnv

file_name = "GGGPolicies/10000furutapendulum"

pendulum = JetsonPendulum(torque_coefficient=0.4)
env = FurutaPendulumEnv(pendulum, steps=1, timestep=240)
env = FrameStack(env, 13)

<<<<<<< HEAD
file_name = "Policies/00004furutapendulum"

=======
# model = SAC('MlpPolicy', env, verbose=1, gamma = 0.95, gradient_steps = 100, train_freq = 1500)
>>>>>>> 6fbc9a4bed4b246d9f87a905f8e65352ed298334
model = SAC.load(file_name)
model.set_env(env)

model.learn(total_timesteps=8000)
<<<<<<< HEAD

model.save("Policies/00005furutapendulum.zip")

=======
model.save("SACPolicies/00004furutapendulum.zip")
>>>>>>> 6fbc9a4bed4b246d9f87a905f8e65352ed298334
env.end()
