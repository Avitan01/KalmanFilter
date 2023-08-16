import numpy as np


class KalmanFilter:
    """Implement a basic Kalman Filter for a linear process"""
    iX = 0
    iV = 1
    NUMVARS = iV + 1
    iA = 0
    iG = 1
    NUMINPS = iG + 1

    def __init__(self,
                 initial_x: float,
                 initial_v: float,
                 initial_a: float,
                 initial_g: float,
                 initial_P: float,
                 accel_variance: float) -> None:
        """Create a Kalman Filter object
            Args:
                initial_x(float): Initial estimation of the x location.
                initial_v(float): Initial estimation of the velocity magnitude.
                initial_a(float): Initial known acceleration profile.
                initial_g(float): Initial known gravity acceleration.
                initial_P(float): The gain for the initial P matrix.
                accel_variance(float): Variance of acceleration disturbance"""
        # Mean of state Gaussian Random Variable
        self._x = np.zeros(self.NUMVARS)
        self._x[self.iX] = initial_x
        self._x[self.iV] = initial_v
        # Input vector
        self._u = np.zeros(self.NUMINPS)
        self._u[self.iA] = initial_a
        self._u[self.iG] = initial_g
        self._Q = 0  # Inputs covariance
        # Covariance of state Gaussian Random Variable
        self._P = initial_P*np.eye(self.NUMVARS)
        self._accel_variance = accel_variance  # Process noise
        self._I = np.eye(self.NUMVARS)
        self._z_innovation = 0

    def predict(self, dt: float) -> None:
        """Predict the next state base on the systems model
            Args:
                dt(float): Time step interval"""
        Phi = np.eye(self.NUMVARS)
        Phi[self.iX, self.iV] = dt  # Transition matrix
        Psi = np.zeros((self.NUMINPS, self.NUMINPS))
        Psi[self.iA] = [0.5 * dt ** 2, -0.5 * dt ** 2]
        Psi[self.iG] = [dt, -dt]
        Gamma = np.zeros((2, 1))  # Process noise matrix
        Gamma[self.iX] = 0.5 * dt ** 2
        Gamma[self.iV] = dt
        self._x = Phi.dot(self._x) + Psi.dot(self._u)
        self._P = Phi.dot(self._P).dot(Phi.T) + Psi.dot(self._Q).dot(Psi.T) + Gamma.dot(Gamma.T) * self._accel_variance

    def update(self, meas_values: float, meas_variance: float, input_update: list = None) -> None:
        """Update the estimation of the state and covariance according to the newly acquired measurements
            Args:
                meas_values(float): The measured values obtained.
                meas_variance(float): The measured variance obtained.
                input_update(list): List of values of input."""
        if input_update:
            for i, val in enumerate(input_update):
                self._u[i] = val
        # self._x/P through the method refers to the estimation of x(k+1|k)
        # self._x/P at the last lines refers to the estimation of x(k+1|k+1)
        H = np.array([1, 0]).reshape((1, 2))  # Observation matrix
        z = np.array([meas_values])  # Measurement
        self._z_innovation = H.dot(self._x)
        R = np.array([meas_variance])  # Variance measured

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        # Josef's formula
        self._P = (self._I - K.dot(H)).dot(self._P).dot((self._I - K.dot(H)).T) + K.dot(R).dot(K)

        self._x = self._x + K.dot(y)

    @property
    def pos(self) -> float:
        return self._x[self.iX]

    @property
    def vel(self) -> float:
        return self._x[self.iV]

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def innovation(self) -> float:
        if isinstance(self._z_innovation, np.ndarray):
            return self._z_innovation[0]
        return self._z_innovation
