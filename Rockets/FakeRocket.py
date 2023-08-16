import numpy as np


class FakeRocket:
    G = 9.81
    FALCON_ACCEL = 5 * 9.81
    TIME_STEP = 0.01
    MAX_SPEED = 9310  # [m/s]

    def __init__(self, initial_height: float, initial_velocity: float, initial_acceleration: float,
                 launch_duration: float, total_duration: float) -> None:
        self._h = initial_height  # [m]
        self._v = initial_velocity  # [m/s]
        self._a = initial_acceleration
        self._launch_duration = launch_duration  # [s]
        self._total_duration = total_duration  # [s]
        self._flight_log = {'h': [], 'v': []}
        self._time = np.arange(0, self._total_duration, self.TIME_STEP)

    def launch(self):
        print('Lift off')
        dt = self.TIME_STEP
        for curr_time in self._time:
            self._flight_log['h'].append(self._h)
            self._flight_log['v'].append(self._v)
            self.acceleration(curr_time)
            self.velocity(dt)
            self.height(dt)

    def update_flight(self, curr_time: float, dt: float) -> None:
        self._flight_log['h'].append(self._h)
        self._flight_log['v'].append(self._v)
        self.acceleration(curr_time)
        self.velocity(dt)
        self.height(dt)

    def height(self, dt) -> None:
        if self._h < 0:
            return
        self._h = self._h + self._v * dt + (0.5 * self._a * dt ** 2 - 0.5 * self.G * dt ** 2)*np.cos(1)

    def velocity(self, dt) -> None:
        self._v = self._v + (self._a * dt - self.G * dt)*np.cos(1)

    def acceleration(self, total_time) -> None:
        if total_time > self._total_duration:
            raise
        if total_time > self._launch_duration:
            self._a = 0
        else:
            self._a = self.FALCON_ACCEL - (total_time / self._total_duration) * self.FALCON_ACCEL
        if self._v > self.MAX_SPEED:
            self._a = 0

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
    def time_vec(self) -> np.array:
        return self._time

    @property
    def time_interval(self) -> float:
        return self.TIME_STEP

    @property
    def flight_duration(self) -> float:
        return self._total_duration

    @property
    def flight_log(self) -> dict:
        return self._flight_log
