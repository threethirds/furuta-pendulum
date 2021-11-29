from math import pi
import random

from pendulum.interface import Pendulum


class MockPendulum(Pendulum):

    def set_rotation(self, rate: float):
        pass

    def angle(self) -> float:
        return random.uniform(-pi, pi)

    def calibrate(self):
        pass
