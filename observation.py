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
    # break out the vehicle state and command (vectorized)
    x, y, theta, _, _, _ = np.atleast_2d(state).T
    _, _, _, xddot, yddot, thetaddot = np.atleast_2d(statedot).T
    # compute expected observations
    accx = xddot
    accy = yddot
    gyro = thetaddot
    mag = theta
    tof = np.sqrt(x**2 + y**2)
    expected_observation = np.stack([accx, accy, gyro, mag, tof]).T
    # and use that to find the likelihood of the given observation
    likelihood = multivariate_normal.pdf(x=(observation-expected_observation),
        cov=simple_2d_observation_cov)
    return np.squeeze(likelihood)
