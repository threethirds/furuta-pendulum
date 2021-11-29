import time
from typing import Dict
from typing import Tuple

import gym
from gym import spaces
import numpy as np

from pendulum.interface import Pendulum


class FurutaPendulumEnv(gym.Env):
    """
    Description:
        Environment for the real Furuta pendulum.

    Observation:
        Type: Box(3)
        Num     Observation                      Min                    Max
        0       Second Arm Sine Angle            -1                      1
        1       Second Arm Cosine Angle          -1                      1
        2       Action                           -1                      1

    Actions:
        Type: Box(1)
        Num     Action          Min                     Max
        0       Torque          -1                       1

    Reward:
        The reward is ((1-(|Second Arm Angle|/π))-0.5)/0.5 each step, the interval is [-0.1, 0.1]
    """

    def __init__(self, pendulum: Pendulum, steps=4, timestep=240):

        self.pendulum = pendulum
        self.steps = steps
        self.timestep_size = 1 / timestep

        self.step_start_time = time.time()
        self.total_amount_of_steps = 0

        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step the pendulum.

        Args:
          action: rotation

        Returns:
            (observation, reward, done, info)
        """
        action = action[0]  # np.clip(action, -1, 1)[0] ??

        inference_duration = time.time() - self.step_start_time
        extra_time = self.timestep_size / 2 - inference_duration
        if extra_time > 0:
            time.sleep(extra_time)

        # set new velocity every self.steps step
        if self.total_amount_of_steps % self.steps == 0:
            self.pendulum.set_rotation(action)

            if self.total_amount_of_steps % 1500 == 1499:
                my_pwm.ChangeDutyCycle(0)

            self.step_counter = 0
        
        # Calculate the current angle 

        armangle = chan.voltage


        if armangle < 0.:
            armangle = self.value_of_zero
        elif armangle>3.33:
            armangle = self.value_of_zero
        elif armangle < self.quarters[0]:
            armangle = -(self.value_of_zero+1)*armangle/self.quarters[0] + self.value_of_zero
        elif armangle<self.quarters[2]:
            armangle = 1-(armangle-self.quarters[0])/1.7
        elif armangle<self.quarters[3]:
            armangle = -(armangle - self.quarters[2])/1.7
        elif armangle<3.33:
            armangle = (armangle-self.quarters[3])*(self.value_of_zero+0.5)/(3.33-self.quarters[3]) - 0.5
        
        alpha = armangle * np.pi

        # get the current angle
        # maybe above velocity ?? (delayed every step steps)
        alpha = self.pendulum.angle()

        total_duration = time.time() - self.step_start_time
        extra_time = self.timestep_size - total_duration
        if extra_time > 0:
            time.sleep(extra_time)

        self.step_start_time = time.time()
        self.total_amount_of_steps += 1

        state = np.array([np.sin(alpha), np.cos(alpha), action], dtype=np.float32)
        reward = (1 - (abs(alpha) / np.pi) - 0.5) / 5
        done = False
        info = {"step_start_time": self.step_start_time, "inference_duration": inference_duration,
                "total_duration": total_duration, "alpha": alpha}

        return state, reward, done, info

    def reset(self):

        # reset the simulation and create from scratch
        self.pendulum.set_rotation(0)
        time.sleep(1)
        self.pendulum.calibrate()

        # Step counter
        self.total_amount_of_steps = 0
        self.step_start_time = time.time()

        return np.array([0., -1., 0.], dtype=np.float32)

    def render(self, mode='human'):
        pass

    def end(self):
        self.pendulum.set_rotation(0)
