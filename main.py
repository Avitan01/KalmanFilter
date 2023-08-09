import matplotlib.pyplot as plt
import matplotlib
import addcopyfighandler
import numpy as np

from KalmanFilter import KalmanFilter as KF
from Rocket import Rocket
matplotlib.use('TkAgg')


fig = plt.figure()
# Simulation parameters
booster_time = 20  # 162
flight_time = booster_time
h0, v0, a0 = 0.0, 0.0, 0.0
kh0, kv0, ka0 = 0.0, 0.0, 0.0
accel_var = 5000
R_covariance = 2 ** 2
MEAS_EVERY_SECONDS = 1

rocket = Rocket(initial_height=h0, initial_velocity=v0, initial_acceleration=a0,
                launch_duration=booster_time, total_duration=flight_time)

kf = KF(initial_x=kh0, initial_v=kv0, initial_a=ka0, initial_g=9.81, accel_variance=accel_var)

mus, covs, real_xs, real_vs = [], [], [], []
DT = rocket.time_interval

for time in rocket.time_vec:
    if time > rocket.flight_duration:
        break
    mus.append(kf.mean)
    covs.append(kf.cov)
    rocket.update_flight(time, DT)
    kf.predict(dt=DT)
    if (time != 0) and (time % MEAS_EVERY_SECONDS) == 0:
        kf.update(meas_values=rocket.pos + np.random.rand() * np.sqrt(R_covariance),
                  meas_variance=R_covariance, input_update=[rocket.accel, 9.81])
        real_xs.append(rocket.pos)
        real_vs.append(rocket.vel)

# Plot estimation
plt.subplot(2, 1, 1)
plt.title('Height', fontsize=18)
plt.plot(rocket.flight_log['h'], 'b')
plt.plot([mu[0] for mu in mus], 'r')
plt.plot([mu[0] - 2 * np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
plt.plot([mu[0] + 2 * np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Height [m]', fontsize=16)

plt.subplot(2, 1, 2)
plt.title('Velocity', fontsize=18)
plt.plot(rocket.flight_log['v'], 'b')
plt.plot([mu[1] for mu in mus], 'r')
plt.plot([mu[1] - 2 * np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
plt.plot([mu[1] + 2 * np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Velocity [m/s]', fontsize=16)

# plt.rcParams['font.size'] = '16'
plt.figlegend(['True value', 'Estimated value', 'MSE+', 'MSE-'], loc='lower right', fontsize=12)
fig.tight_layout()
plt.show()
