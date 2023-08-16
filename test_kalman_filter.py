import numpy as np
import unittest

from KalmanFilter import KalmanFilter as KF


class TestKF(unittest.TestCase):
    """Test the Kalman filter implemented"""
    def test_can_construct_with_x_and_v(self):
        x = 0.2
        v = 2.3
        a = 0
        g = 9.81
        P = 100

        kf = KF(initial_x=x, initial_v=v, initial_a=a, initial_g=g, initial_P=P , accel_variance=1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)

    def test_after_calling_predict_mean_and_cov_are_of_right_shape(self):
        x = 0.2
        v = 2.3
        a = 0
        g = 9.81
        P = 100

        kf = KF(initial_x=x, initial_v=v, initial_a=a, initial_g=g, initial_P=P, accel_variance=1.2)
        kf.predict(dt=0.1)

        self.assertEqual(kf.mean.shape, (2, ))
        self.assertEqual(kf.cov.shape, (2, 2))

    def test_calling_predict_increases_state_uncertainty(self):
        x = 0.2
        v = 2.3
        a = 0
        g = 9.81
        P = 100

        kf = KF(initial_x=x, initial_v=v, initial_a=a, initial_g=g, initial_P=P, accel_variance=1.2)
        kf.predict(dt=0.1)

        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt=0.1)
            det_after = np.linalg.det(kf.cov)

            self.assertGreater(det_after, det_before)

    def test_calling_update_decreases_state_uncertainty(self):
        x = 0.2
        v = 2.3
        a = 0
        g = 9.81
        P = 100

        kf = KF(initial_x=x, initial_v=v, initial_a=a, initial_g=g, initial_P=P, accel_variance=1.2)
        det_before = np.linalg.det(kf.cov)
        kf.update(meas_values=0.1, meas_variance=0.1)
        det_after = np.linalg.det(kf.cov)

        self.assertLess(det_after, det_before)
        