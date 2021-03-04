import numpy as np
from dynamics import simple_2d_dynamics

def simple_2d_acc_gyro_mag(state, command):
	acc_noise_scale = 0.5  # m/s/s
	gyro_noise_scale = 0.1  # radians/s
	mag_noise_scale = 0.1  # radians/s

	# we will need the accelerations as well
	statedot = simple_2d_dynamics(state, command)

	# compute simulated observations
	acc = state_dot[3:5] + np.random.randn(2) * acc_noise_scale
	gyro = state_dot[5] + np.random.randn(1) * gyro_noise_scale
	mag = state[0:2] + np.random.randn(2) * mag_noise_scale

	# return the vector of observations
	return np.concatentate([acc, gyro, mag])
