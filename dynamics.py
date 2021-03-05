import numpy as np


# x, y, theta, xdot, ydot, thetadot
simple_2d_transition_cov = np.diag([0.01, 0.01, 0.005, 0.001, 0.001, 0.0005])

def simple_2d_dynamics(state, command):
    # break out the vehicle state and command (vectorized)
    x, y, theta, xdot, ydot, thetadot = np.atleast_2d(state).T
    l_thrust, r_thrust = np.atleast_2d(command).T

    # compute the accelerations (including linear drag)
    xddot = np.cos(theta) * (r_thrust + l_thrust) - xdot * np.abs(xdot) * 0.95
    yddot = np.sin(theta) * (r_thrust + l_thrust) - ydot * np.abs(ydot) * 0.9
    thetaddot = r_thrust - l_thrust - thetadot * np.abs(thetadot) * 0.95

    # and return the time derivative of the state
    statedot = np.stack([xdot, ydot, thetadot, xddot, yddot, thetaddot]).T
    return np.squeeze(statedot)
