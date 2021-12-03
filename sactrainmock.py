import time

from gym.wrappers import FrameStack
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from pendulum.mock import MockPendulum
from real import FurutaPendulumEnv

pendulum = MockPendulum()
env = FurutaPendulumEnv(pendulum, steps=1, timestep=240)
env = FrameStack(env, 13)

model = SAC('MlpPolicy', env, verbose=1, gamma=0.95, gradient_steps=100, train_freq=1500, device='cuda')
model.set_env(env)


class CB(BaseCallback):

    def __init__(self):
        super().__init__()
        self.start = None

    def _on_step(self) -> bool:
        pass

    def on_training_start(self, *args):
        self.start = time.time()
        print('start')

    def on_training_end(self):
        print('end')
        print(time.time() - self.start)


model.learn(total_timesteps=8000, callback=CB())
env.end()
