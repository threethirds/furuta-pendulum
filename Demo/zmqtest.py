import itertools as it
import math
import zmq

import time
import busio
from board import SDA, SCL
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import Jetson.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
import gym
from gym import spaces
import numpy as np
from os import path

# Create the I2C bus
i2c = busio.I2C("GEN2_I2C_SCL","GEN2_I2C_SDA")
# Create the ADC object using the I2C bus
ads = ADS.ADS1015(i2c)
# Create single-ended input on channel 0
chan = AnalogIn(ads, ADS.P0)
#print("{:>5}\t{:>5}".format('raw', 'v'))

x = chan.voltage
quarters = [x, x+0.86, x+1.72, x+2.56]

for i in range(len(quarters)):
    quarters[i] = quarters[i]%3.33
zero_loc = 0
for i in range(len(quarters)-1):
    zero = quarters[i+1] - quarters[i]
    if(zero<0): 
        zero_loc = i+1

angle_of_zero = (zero_loc+((0.85-quarters[zero_loc])/0.85))/4

value_of_zero = -0.5

if angle_of_zero < 0:
    value_of_zero = -0.5
elif angle_of_zero > 1:
    value_of_zero = -0.5
elif angle_of_zero < 0.25:
    value_of_zero = -(angle_of_zero*2 + 0.5)
elif angle_of_zero <0.75:
    value_of_zero = 1-((angle_of_zero-0.25)*2)
elif angle_of_zero < 1:
    value_of_zero = -((angle_of_zero-0.75)*2)
print(quarters, value_of_zero)

# Create socket
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind(f"tcp://*:3333")

for i in it.count():
    armangle = chan.voltage
    
    if armangle < 0.:
        armangle = value_of_zero
    elif armangle>3.33:
        armangle = value_of_zero
    elif armangle < quarters[0]:
        armangle = -(value_of_zero+1)*armangle/quarters[0] + value_of_zero
    elif armangle<quarters[2]:
        armangle = 1-(armangle-quarters[0])/1.7
    elif armangle<quarters[3]:
        armangle = -(armangle - quarters[2])/1.7
    elif armangle<3.33:
        armangle = (armangle-quarters[3])*(value_of_zero+0.5)/(3.33-quarters[3]) - 0.5
    alpha = armangle * np.pi

    sine = np.sin(alpha)
    cosine = np.cos(alpha)

    reward = 1 - (abs(alpha)/np.pi)
    new_reward = (reward-0.5)/5
    #print(chan.voltage, armangle, new_reward)
    data = {'timestamp': time.time(),
            'voltage': chan.voltage,
            'armangle': armangle,
            'sine': sine,
            'cosine': cosine,
            'reward': new_reward}
    socket.send_json(data)
    time.sleep(1./240)
