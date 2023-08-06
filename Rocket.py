import numpy as np


class Rocket:
    G = 9.81
    FALCON_ACCEL = 5 * 9.81
    TIME_SEG = 1000

    def __init__(self, initial_height: float, initial_velocity: float, initial_acceleration: float,
                 launch_duration: float, total_duration: float) -> None:
        self._h = initial_height  # [m]
        self._v = initial_velocity  # [m/s]
        self._a = initial_acceleration
        self._launch_duration = launch_duration  # [s]
        self._total_duration = total_duration  # [s]
        self._flight_log = []
        self.time = np.linspace(0, self._total_duration, self.TIME_SEG)

    def launch(self):
        print('Lift off')
        dt = self._total_duration / self.TIME_SEG
        for curr_time in self.time:
            self._flight_log.append(self._h)
            self.acceleration(curr_time)
            self.velocity(dt)
            self.height(dt)

    def height(self, dt) -> None:
        if self._h < 0:
            return
        self._h = self._h + self._v * dt + 0.5 * self._a * dt ** 2 - 0.5 * self.G * dt ** 2

    def velocity(self, dt) -> None:
        self._v = self._v + self._a * dt - self.G * dt

    def acceleration(self, total_time) -> None:
        if total_time > self._total_duration:
            raise
        if total_time > self._launch_duration:
            self._a = 0
        else:
            self._a = self.FALCON_ACCEL

    @property
    def pos(self) -> float:
        return self._h

    @property
    def vel(self) -> float:
        return self._v

    @property
    def accel(self) -> float:
        return self._a

    @property
    def flight_log(self) -> list:
        return self._flight_log
