import numpy as np
from scipy.stats import multivariate_normal


# accx, accy, gyro, mag, tof
simple_2d_observation_cov = np.diag([0.05, 0.05, 0.001, 0.01, 0.1])

# for simulation
def simple_2d_H_sample(state, statedot, command):
    # compute simulated observations
    accx = statedot[3]
    accy = statedot[4]
    gyro = statedot[5]
    mag = state[2]
    tof = np.linalg.norm(state[:2])
    observation = np.array([accx, accy, gyro, mag, tof])
    # and additive noise
    noise = multivariate_normal.rvs(cov=simple_2d_observation_cov)
    return observation + noise

# for state estimation
def simple_2d_H_likelihood(state, statedot, command, observation):
    # compute expected observations
    accx = statedot[3]
    accy = statedot[4]
    gyro = statedot[5]
    mag = state[2]
    tof = np.linalg.norm(state[:2])
    expected_observation = np.array([accx, accy, gyro, mag, tof])
    # and use that to find the likelihood of the given observation
    likelihood =  multivariate_normal.pdf(x=observation,
        mean=expected_observation, cov=simple_2d_observation_cov)
    return likelihood
