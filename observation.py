import numpy as np
from scipy.stats import multivariate_normal

from dynamics import simple_2d_dynamics


# accx, accy, gyro, mag
simple_2d_observation_cov = np.diag([0.5, 0.5, 0.1, 0.1,])

# for simulation
def simple_2d_acc_gyro_mag_sample(state, command):
    # we will need the accelerations as well
    statedot = simple_2d_dynamics(state, command)
    # compute simulated observations
    accx = statedot[3]
    accy = statedot[4]
    gyro = statedot[5]
    mag = state[2]
    observation = np.array([accx, accy, gyro, mag])
    # and additive noise
    noise = multivariate_normal.rvs(cov=simple_2d_observation_cov)
    return observation + noise

# for state estimation
def simple_2d_acc_gyro_mag_likelihood(state, command, observation):
    # we will need the accelerations as well
    statedot = simple_2d_dynamics(state, command)
    # compute expected observations
    accx = statedot[3]
    accy = statedot[4]
    gyro = statedot[5]
    mag = state[2]
    expected_observation = np.array([accx, accy, gyro, mag])
    # and use that to find the likelihood of the given observation
    likelihood =  multivariate_normal.pdf(x=observation,
        mean=expected_observation, cov=simple_2d_observation_cov)
    return likelihood
