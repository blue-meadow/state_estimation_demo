import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation as R

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


def simple_3d_dynamics(x, u):
    pos, rot, vel, ang_vel = np.split(x, [3, 7, 10])
    thrust, pitch, yaw = u
    rot = R.from_quat(rot)

    # assume a small roll stabilization force gets generated (by buoyancy)
    roll = -0.3 * rot.apply([0,1,0]).dot([0,0,1])

    # compute forces from drag and from thrusters
    drag = 1. * vel**2 * np.sign(vel)
    acc = rot.apply([thrust, 0, 0]) - drag

    ang_drag = 5. * ang_vel**2 * np.sign(ang_vel)
    ang_acc = rot.apply([roll, pitch, yaw]) - ang_drag

    return np.concatenate([vel, ang_vel, acc, ang_acc])


def casadi_3d_dynamics(x, u):
    # x, y, z, r, p, y (meters)
    thrust_poses = np.array([
        [-0.2, 0.1, 0, 0, 0, 0],        # left
        [-0.2, -0.1, 0, 0, 0, 0],       # right
        [-0.1, 0, 0, 0, -np.pi/2, 0]])   # vertical

    mass = 1.0  # kg

    # break out the state vector
    pos = x[:3]
    rot = x[3:7]
    vel = x[7:10]
    ang_vel = x[10:]


    # compute forces and torques resulting from control
    force_vecs = R.from_euler('xyz', thrust_poses[:, 3:]).apply([1, 0, 0])
    torque_vecs = np.cross(thrust_poses[:, :3], force_vecs)
    thrust_forces = u * force_vecs
    thrust_torques = u * torque_vecs


    # compute forces and torques resulting from drag and buoyancy
    # TODO

    acc = ca.transpose(ca.sum1(thrust_forces))/mass
    ang_acc = ca.transpose(ca.sum1(thrust_forces))/mass  # NOTE: should be inertial tensor
    ang_acc = R.from_rotvec(ang_acc).as_euler('xyz')

    return ca.vertcat(vel, ang_vel, acc, ang_acc)

    
