from numpy.lib.function_base import angle
import gym
from gym import spaces
import time
import busio
from board import SDA, SCL
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import Jetson.GPIO as GPIO
import numpy as np
from os import path
from collections import deque

GPIO.setmode(GPIO.BCM)
# Create the I2C bus
i2c = busio.I2C("GEN2_I2C_SCL","GEN2_I2C_SDA")
# Create the ADC object using the I2C bus
ads = ADS.ADS1015(i2c)
# Create single-ended input on channel 0
chan = AnalogIn(ads, ADS.P0)
time.sleep(1)
#PWM
GPIO.setup(13, GPIO.OUT)
GPIO.setup(20, GPIO.OUT, initial=1)
my_pwm = GPIO.PWM(13, 10000)



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
        The reward is ((1-(|Second Arm Angle|/3.1415))-0.5)/0.5 each step, the interval is [-0.1, 0.1]

    Starting State:
        All observations have values 0.

    Episode Termination:
        Episode length is greater than episode_termination_variable.

    Solved Requirements:
        Will be concluded later
    """
    def __init__(self):

        # set values
        #self.max_torque = 100.0

        self.state = None

        self.actual_action_size = 0

        # Initialize start times

        self.step_start_time = time.time()

        # Initialize step counters

        self.step_counter = 0

        self.total_amount_of_steps = 0

        # set observation and action spaces

        high = np.array([1,
                         1,
                         1], 
                        dtype=np.float32) #observation space
        
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1,
            high=1, shape=(1,), 
            dtype=np.float32
        )
    
    def variable_init(self, steps = 4, coefficient = 1, timestep = 240):
        self.steps = steps
        self.torque_coefficient = coefficient
        self.timestep_size = 1./timestep

    def step(self, action, done=False):
        """Step forward the simulation, given the action.

        Args:
          action: Value of the torque

        Returns:
          observations: Observation space
          reward: Normalized magnitude of the angle of the second arm
          done: For now only False.
          info: For the moment this will be empty

        Raises:
          Errors
        """
        self.predict_end_time = time.time()

        step_predict_time = self.predict_end_time-self.step_start_time

        faster_than_expected = self.timestep_size/2 - step_predict_time
        if faster_than_expected>0:
            time.sleep(faster_than_expected)
            step_predict_time = self.timestep_size/2

        action = np.clip(action, -1, 1)[0]

        # Calculate new velocity every self.steps step

        if self.step_counter%self.steps == 0:

            #  use the action on the simulation

            self.actual_action_size = action
            
            direction = np.sign(action)
            if direction < 0: 
                direction = 0

            
            adjusted_action = abs(action) * 100.0 * self.torque_coefficient

            GPIO.output(20, direction)
            my_pwm.ChangeDutyCycle(adjusted_action)

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

        sine = np.sin(alpha)
        cosine = np.cos(alpha)

        self.state = (sine, cosine, action)

        # calculate the reward


        reward = 1 - (abs(alpha)/np.pi)
        new_reward = (reward-0.5)/5

        self.step_counter += 1

        self.total_amount_of_steps += 1
        
        #trying to make the frequency 120Hz

        self.step_end_time = time.time()

        step_total_time = self.step_end_time-self.step_start_time

        faster_than_expected = self.timestep_size - step_total_time

        if faster_than_expected>0:
            time.sleep(faster_than_expected)
            step_total_time = self.timestep_size

        self.step_start_time = time.time()

        return np.array(self.state, dtype=np.float32), new_reward, done, {"step2": step_predict_time, "alpha":alpha, "actual action":self.actual_action_size, "sine": sine, "cosine": cosine, "reward": new_reward, "action": action, "step time": step_total_time}

    def reset(self):
        my_pwm.start(0)
        # reset the simulation and create from scratch
        time.sleep(1)

        # Step counter

        self.step_counter = 0

        self.total_amount_of_steps = 0

        self.actual_action_size = 0

        # Set up the potenciometer
        x = chan.voltage
        if x<0. or x>0.78:
            return "change potenciometer hand location"
        self.quarters = [x, x+0.86, x+1.72, x+2.56]        
        for i in range(len(self.quarters)):
            self.quarters[i] = self.quarters[i]%3.33
        zero_loc = 0     
        angle_of_zero = ((0.85-self.quarters[zero_loc])/0.85)/4        
        self.value_of_zero = -0.5        
        if angle_of_zero < 0:
            self.value_of_zero = -0.5
        elif angle_of_zero > 1:
            self.value_of_zero = -0.5
        elif angle_of_zero < 0.25:
            self.value_of_zero = -(angle_of_zero*2 + 0.5)

        self.step_start_time = time.time() 

        self.state = (0., -1., 0.)
            
        return np.array(self.state, dtype=np.float32)
    
    def _render(self):
      return

    def end(self):
        my_pwm.ChangeDutyCycle(0)
        GPIO.cleanup()
        return

