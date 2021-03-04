import numpy as np

def simple_2d_dynamics(state, command):
	# break out the vehicle state and command
	x, y, theta, xdot, ydot, thetadot = state
	l_thrust, r_thrust = command

	# compute the accelerations
	xddot = np.cos(theta) * (r_thrust + l_thrust)
	xddot = np.sin(theta) * (r_thrust + l_thrust)
	thetaddot = r_thrust - l_thrust

	# and return the time derivative of the state
	return np.array([xdot, ydot, thetadot, xddot, yddot, thetaddot])
