import atexit
import math

import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import busio
import Jetson.GPIO as GPIO

from pendulum.interface import Pendulum


class JetsonPendulum(Pendulum):

    def __init__(self, torque_coefficient):

        self.torque_coefficient = torque_coefficient
        GPIO.setmode(GPIO.BCM)

        # ADC
        i2c = busio.I2C("GEN2_I2C_SCL", "GEN2_I2C_SDA")
        ads = ADS.ADS1015(i2c)
        self.adc = AnalogIn(ads, ADS.P0)

        # PWM
        GPIO.setup(13, GPIO.OUT)
        GPIO.setup(20, GPIO.OUT, initial=1)
        self.pwm = GPIO.PWM(13, 10000)
        self.pwm.start(0)

        self.quarters = None
        self.value_of_zero = None
        atexit.register(GPIO.cleanup)
        atexit.register(lambda: self.set_rotation(0))

    def set_rotation(self, rate: float):

        assert -1 <= rate <= 1

        direction = 0 if rate < 0 else 1
        duty_cycle = abs(rate) * 100.0 * self.torque_coefficient

        GPIO.output(20, direction)
        self.pwm.ChangeDutyCycle(duty_cycle)

    def angle(self) -> float:

        voltage = self.adc.voltage

        if voltage < 0.:
            arm_angle = self.value_of_zero
        elif voltage < self.quarters[0]:
            arm_angle = -(self.value_of_zero + 1) * voltage / self.quarters[0] + self.value_of_zero
        elif voltage < self.quarters[2]:
            arm_angle = 1 - (voltage - self.quarters[0]) / 1.7
        elif voltage < self.quarters[3]:
            arm_angle = -(voltage - self.quarters[2]) / 1.7
        elif voltage < 3.33:
            arm_angle = (voltage - self.quarters[3]) * (self.value_of_zero + 0.5) / (3.33 - self.quarters[3]) - 0.5
        else:
            arm_angle = self.value_of_zero

        return arm_angle * math.pi

    def calibrate(self):

        # assume pointing down
        x = self.adc.voltage

        if x < 0. or x > 0.78:
            raise EnvironmentError("change potentiometer hand location")

        self.quarters = [x, x + 0.86, x + 1.72, x + 2.56]
        for i in range(len(self.quarters)):
            self.quarters[i] = self.quarters[i] % 3.33

        angle_of_zero = (0.85 - x) / 0.85

        self.value_of_zero = -(angle_of_zero / 2 + 0.5)
