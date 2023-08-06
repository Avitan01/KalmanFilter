import matplotlib.pyplot as plt
import numpy as np

from KalmanFilter import KalmanFilter as KF
from Rocket import Rocket

plt.ion()  # Enable interactive mode
plt.figure()

rocket = Rocket(initial_height=0.0, initial_velocity=0.0, initial_acceleration=0.0,
                launch_duration=162, total_duration=800)

rocket.launch()
# print(rocket.flight_log)
# plt.plot(rocket.time, rocket.flight_log)

# # Real Process
# real_x = 0.0
# meas_variance = 0.1**2
# real_v = 0.9
#
kf = KF(initial_x=0.0, initial_v=0.0, accel_variance=0.1)
#
# DT = 0.1
# NUM_STEPS = 1000
# MEAS_EVERY_STEPS = 20
#
# mus, covs, real_xs, real_vs = [], [], [], []
#
# for step in range(NUM_STEPS):
#     mus.append(kf.mean)
#     covs.append(kf.cov)
#     real_x += DT * real_v
#     kf.predict(dt=DT)
#     if (step != 0) and (step % MEAS_EVERY_STEPS) == 0:
#         kf.update(meas_values=real_x + np.random.rand() * np.sqrt(meas_variance),
#                   meas_variance=meas_variance)
#     real_xs.append(real_x)
#     real_vs.append(real_v)
#
# # Plot estimation
# plt.subplot(2, 1, 1)
# plt.title('Position')
# plt.plot([mu[0] for mu in mus], 'r')
# plt.plot(real_xs, 'b')
# plt.plot([mu[0] - 2*np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
# plt.plot([mu[0] + 2*np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
#
# plt.subplot(2, 1, 2)
# plt.title('Velocity')
# plt.plot([mu[1] for mu in mus], 'r')
# plt.plot(real_vs, 'b')
# plt.plot([mu[1] - 2*np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
# plt.plot([mu[1] + 2*np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
#
plt.show()
plt.ginput(-1)




