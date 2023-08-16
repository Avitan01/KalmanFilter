import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from Rockets.Rocket import Rocket
from KalmanFilter import KalmanFilter as KF

matplotlib.use('TkAgg')


def monte_carlo(N: float) -> dict:
    # Define constant parameters
    booster_time = 20  # 162
    flight_time = booster_time + 20
    h0, v0, a0 = 0.0, 0.0, 0.0

    P0 = 1000
    accel_var = 0.5
    R_covariance = 2 ** 2
    MEAS_EVERY_STEPS = 50
    height_estimation_error = np.zeros((N, 4000))
    velocity_estimation_error = np.zeros((N, 4000))
    for j in range(N):
        rocket = Rocket(initial_height=h0, initial_velocity=v0, initial_acceleration=a0,
                        launch_duration=booster_time, total_duration=flight_time)
        kh0, kv0, ka0 = 0.0, 0.0, 0.0
        kf = KF(initial_x=kh0, initial_v=kv0, initial_a=ka0,
                initial_g=9.81, initial_P=P0, accel_variance=accel_var)

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
                kf.update(meas_values=rocket.pos + np.random.rand() * np.sqrt(R_covariance),
                          meas_variance=R_covariance, input_update=[rocket.accel, 9.81])
            real_xs.append(rocket.pos)
            real_vs.append(rocket.vel)
            height_estimation_error[j, i] = rocket.pos - kf.pos
            velocity_estimation_error[j, i] = rocket.vel - kf.vel
    return {'height estimation error': height_estimation_error,
            'velocity estimation error': velocity_estimation_error,
            'time vector': rocket.time_vec,
            'covariance matrix': covs}


if __name__ == '__main__':
    monte_carlo_dict = monte_carlo(1000)
    height_estimation_error = monte_carlo_dict['height estimation error']
    velocity_estimation_error = monte_carlo_dict['velocity estimation error']
    time_vec = monte_carlo_dict['time vector']
    covs = monte_carlo_dict['covariance matrix']
    fig = plt.figure()
    mean = np.mean(height_estimation_error, axis=0)
    stds = []
    for est_error in height_estimation_error:
        plt.plot(time_vec, est_error)
    plt.title('Monte Carlo simulation N=1000', fontsize=16)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Height [m]', fontsize=16)
    #
    fig2 = plt.figure()
    meanv = np.mean(velocity_estimation_error, axis=0)
    for est_error in velocity_estimation_error:
        plt.plot(time_vec, est_error)
    plt.title('Monte Carlo simulation N=1000', fontsize=16)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Velocity [m/s]', fontsize=16)
    #
    height_std = np.std(height_estimation_error, axis=0)
    fig3 = plt.figure()
    plt.plot(time_vec, mean)
    plt.plot(time_vec, np.add(height_std, mean), 'r--')
    plt.plot(time_vec, np.subtract(mean, height_std), 'r--')
    plt.title('Monte Carlo simulation N=1000', fontsize=16)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Mean estimation error Height', fontsize=16)
    plt.legend(['$\mu$', '$\sigma$'])

    velocity_std = np.std(velocity_estimation_error, axis=0)
    fig4 = plt.figure()
    plt.plot(time_vec, meanv)
    plt.plot(time_vec, np.add(meanv, velocity_std), 'r--')
    plt.plot(time_vec, np.subtract(meanv, velocity_std), 'r--')
    plt.title('Monte Carlo simulation N=1000', fontsize=16)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Mean estimation error Velocity', fontsize=16)
    plt.legend(['$\mu$', '$\sigma$'])

    fig5 = plt.figure()
    plt.plot(time_vec, height_std, 'r--')
    plt.plot(time_vec, -height_std, 'r--')
    plt.plot(time_vec, [np.sqrt(cov[0, 0]) for cov in covs], 'b')
    plt.plot(time_vec, [-np.sqrt(cov[0, 0]) for cov in covs], 'b')
    plt.title('Monte Carlo simulation N=1000', fontsize=16)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Height STD', fontsize=16)
    plt.legend(['STD calc', 'STD calc', 'STD filter', 'STD filter'])

    fig6 = plt.figure()
    plt.plot(time_vec, velocity_std, 'r--')
    plt.plot(time_vec, -velocity_std, 'r--')
    plt.plot(time_vec, [np.sqrt(cov[1, 1]) for cov in covs], 'b')
    plt.plot(time_vec, [-np.sqrt(cov[1, 1]) for cov in covs], 'b')
    plt.title('Monte Carlo simulation N=1000', fontsize=16)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Velocity STD', fontsize=16)
    plt.legend(['STD calc', 'STD calc', 'STD filter', 'STD filter'])
    plt.show()
