import abc


class Pendulum(abc.ABC):

    @abc.abstractmethod
    def set_rotation(self, rate: float):
        """Set desired rotation rate

        Args:
            rate: real number between -1 and 1. Exact physical meaning is undefined,
            but 1 means fast, 0 stand still, and -1 fast in a different direction.
        """

    @abc.abstractmethod
    def angle(self) -> float:
        """Get current angle

        Returns:
            current angle in radians between -π and π, where 0 means pointing up.
        """

    @abc.abstractmethod
    def calibrate(self):
        """Calibrate the pendulum controller"""
