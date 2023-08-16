import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from KalmanFilter import KalmanFilter as KF
from Rockets.Rocket import Rocket
from Rockets.FakeRocket import FakeRocket

matplotlib.use('TkAgg')

fig = plt.figure()
# Simulation parameters
booster_time = 20  # 162
flight_time = booster_time + 20
h0, v0, a0 = 0.0, 0.0, 0.0
kh0, kv0, ka0 = 0.0, 0.0, 0.0
P0 = 1000
accel_var = 0.5
R_covariance = 2 ** 2
MEAS_EVERY_STEPS = 50
delay = 1

rocket_type = 'real'
if rocket_type == 'real':
    rocket = Rocket(initial_height=h0, initial_velocity=v0, initial_acceleration=a0,
                    launch_duration=booster_time, total_duration=flight_time)
else:
    rocket = FakeRocket(initial_height=h0, initial_velocity=v0, initial_acceleration=a0,
                        launch_duration=booster_time, total_duration=flight_time)

kf = KF(initial_x=kh0, initial_v=kv0, initial_a=ka0, initial_g=9.81, initial_P=P0, accel_variance=accel_var)

mus, covs, real_xs, real_vs, innovation, estimation_error = [], [], [], [], [], {'h': [], 'v': []}
DT = rocket.time_interval

for i, time in enumerate(rocket.time_vec):
    if time > rocket.flight_duration:
        break
    mus.append(kf.mean)
    covs.append(kf.cov)
    innovation.append(kf.innovation)
    rocket.update_flight(time, DT)
    kf.predict(dt=DT)
    if (i != 0) and (i % MEAS_EVERY_STEPS) == 0:
        if delay:
            time_delay = -int(delay // rocket.TIME_STEP)
            if len(rocket.flight_log['h']) < abs(time_delay):
                position = rocket.flight_log['h'][0]
            else:
                position = rocket.flight_log['h'][time_delay]
            kf.update(meas_values=position + np.random.rand() * np.sqrt(R_covariance),
                      meas_variance=R_covariance, input_update=[rocket.accel, 9.81])
        else:
            kf.update(meas_values=rocket.pos + np.random.rand() * np.sqrt(R_covariance),
                      meas_variance=R_covariance, input_update=[rocket.accel, 9.81])
    real_xs.append(rocket.pos)
    real_vs.append(rocket.vel)
    estimation_error['h'].append(rocket.pos - kf.pos)
    estimation_error['v'].append(rocket.vel - kf.vel)

# Plot estimation
plt.subplot(1, 2, 1)
plt.title('Height', fontsize=18)
plt.plot(rocket.time_vec, rocket.flight_log['h'], 'b')
plt.plot(rocket.time_vec, [mu[0] for mu in mus], 'r--')
# plt.plot(rocket.time_vec, [mu[0] - 2 * np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
# plt.plot(rocket.time_vec, [mu[0] + 2 * np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Height [m]', fontsize=16)

plt.subplot(1, 2, 2)
plt.title('Velocity', fontsize=18)
plt.plot(rocket.time_vec, rocket.flight_log['v'], 'b')
plt.plot(rocket.time_vec, [mu[1] for mu in mus], 'r')
# plt.plot(rocket.time_vec, [mu[1] - 2 * np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
# plt.plot(rocket.time_vec, [mu[1] + 2 * np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Velocity [m/s]', fontsize=16)

# plt.rcParams['font.size'] = '16'
plt.figlegend(['True value', 'Estimated value'], loc='lower right', fontsize=12)
fig.tight_layout()

fig2 = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(rocket.time_vec, estimation_error['h'])
plt.plot(rocket.time_vec, [np.sqrt(cov[0, 0]) for cov in covs], 'r--')
plt.plot(rocket.time_vec, [-np.sqrt(cov[0, 0]) for cov in covs], 'r--')
plt.title(f'Height Estimation error - R ={np.sqrt(R_covariance)}^2')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('STD', fontsize=16)

plt.subplot(1, 2, 2)
plt.plot(rocket.time_vec, estimation_error['v'])
plt.plot(rocket.time_vec, [np.sqrt(cov[1, 1]) for cov in covs], 'r--')
plt.plot(rocket.time_vec, [-np.sqrt(cov[1, 1]) for cov in covs], 'r--')
plt.title(f'Velocity Estimation error - R ={np.sqrt(R_covariance)}^2')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('STD', fontsize=16)
plt.figlegend(['Estimation error', '$\sigma$'], loc='lower right', fontsize=12)

fig3 = plt.figure()
plt.plot(rocket.time_vec, rocket.flight_log['h'], 'b')
plt.plot(rocket.time_vec, innovation, 'r--')
plt.title('Innovation comparison', fontsize=18)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Height [m]', fontsize=16)
plt.legend(['True location', 'Innovation'])
plt.show()
