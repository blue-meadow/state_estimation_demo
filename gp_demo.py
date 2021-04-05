import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy as np
from perlin_noise import PerlinNoise
from scipy.spatial.transform import Rotation as R
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

###############################################################################
# SIMULATION
###############################################################################

def dynamics(x, u):
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

def euler_step(x, u, dt):
    # apply the dynamics update and break out the state vectors
    x_d = dynamics(x, u)
    pos, rot, vel, ang_vel = np.split(x, [3, 7, 10])
    vel_, ang_vel_, acc_, ang_acc_ = np.split(x_d * dt, [3, 6, 9])

    # compute the next state. only hard part is rotation
    pos_new = pos + vel_
    rot_new = (R.from_euler('xyz', ang_vel_) * R.from_quat(rot)).as_quat()
    vel_new = vel + acc_
    ang_vel_new = ang_vel + ang_acc_

    return np.concatenate([pos_new, rot_new, vel_new, ang_vel_new])

def point_and_shoot_controller(x, wp):
    pos, rot, vel, ang_vel = np.split(x, [3, 7, 10])
    rot = R.from_quat(rot)
    to_wp = wp - pos

    # positive thrust if we are facing toward the waypoint
    heading_vec = rot.apply([1, 0, 0])
    alignment = heading_vec.dot(to_wp/np.linalg.norm(to_wp))
    thrust = 0.1 * alignment * (alignment > 0.5)

    # proportional control to face the waypoint
    e_x, e_y, e_z = rot.inv().apply(to_wp)
    e_pitch = -np.arctan2(e_z, e_x)
    e_yaw = np.arctan2(e_y, e_x)
    pitch = 2 * e_pitch * (np.abs(e_yaw) < np.pi/2)
    yaw = 2 * e_yaw

    return np.array([thrust, pitch, yaw])

def simulate(x, controller, H, dt=0.01):
    x_log = [x]
    u_log = []
    ctx_log = []

    for h in range(H):
        u = controller(x)
        x = euler_step(x, u, dt)

        x_log.append(x)
        u_log.append(u)

        if isinstance(controller, WPMission):
            ctx_log.append(controller.wp)

    return np.array(x_log), np.array(u_log), np.array(ctx_log)


class WPMission:
    def __init__(self):
        self.new_wp()

    def new_wp(self):
        self.wp = np.random.rand(3) * np.array([10, 10, 5]) - 5
        print('Set WP to:', self.wp)

    def __call__(self, x):
        if np.linalg.norm(x[:3] - self.wp) < 0.5: self.new_wp()
        return point_and_shoot_controller(x, self.wp)

###############################################################################
# VISUALIZATION
###############################################################################

def plot_axis(ax, pos, quat, scale):
    ends = pos + R.from_quat(quat).apply(np.eye(3) * scale)
    for end, c in zip(ends, ['r', 'g', 'b']):
        ax.plot3D(*np.stack([pos, end]).T, c=c)

def animate(X, U, stepsize=1, wps = None):
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    def update(num):
        i = num * stepsize
        # NOTE(izzy): there is a way to just update the lines in the plot
        # without clearing everything... i'll figure that out eventually
        ax.cla()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 0)
        # plot the line p to this point
        ax.plot3D(*X[:max(i, 2), 0:3].T, c='k')
        # plot the axis
        plot_axis(ax, X[i,0:3], X[i,3:7], 0.25)
        # and plot a waypoint
        if wps is not None: ax.scatter(*wps[i])

    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update, X.shape[0]//stepsize,
                              interval=30, blit=False)
    plt.show()

def get_slice(f, res, z=0):
    img = np.zeros([res, res])
    for i, x in enumerate(np.linspace(-5, 5, res)):
        for j, y in enumerate(np.linspace(-5, 5, res)):
            img[j, i] = f([x, y, z])

    return img

def get_slice_vectorized(f, res, z):
    xs, ys = np.meshgrid(np.linspace(-5, 5, res), np.linspace(-5, 5, res))
    zs = np.ones([res, res]) * z
    coords = np.stack([xs, ys, zs]).reshape([3, -1]).T
    return f(coords).reshape([res, res])

###############################################################################
# ESTIMATION
###############################################################################

if __name__ == '__main__':
    # create initial state
    pos = np.array([0, 0, -2])
    rot = R.from_euler('xyz', np.radians([0, 0, 0])).as_quat()
    vel = np.zeros(3)
    ang_vel = np.array([0, 0, 0])
    x = np.concatenate([pos, rot, vel, ang_vel])

    print('Simulating a random waypoint mission.')
    controller = WPMission()
    X, U, ctx = simulate(x, controller, 50000)
    print('Press q to continue.')
    animate(X, U, stepsize=50, wps=ctx)

    print('Simulating observations from the map.')
    env = PerlinNoise(octaves=0.3)  # serve as concentration map
    positions = X[::max(1, X.shape[0]//1000), :3] # subsample to ~1000 points
    observations = np.array([env(p) for p in positions])

    print('Gaussian process regression.')
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel,
            random_state=0).fit(positions, observations)

    print('Generating comparison images.')
    ground_truth_map = get_slice(env, 100, z=-2)
    estimated_map = get_slice_vectorized(gpr.predict, 100, z=-2)
    standard_deviation_map = get_slice_vectorized(
        lambda x: gpr.predict(x, return_std=True)[1], 100, z=-2)
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(ground_truth_map)
    axs[0].set_title('Ground Truth')
    axs[1].imshow(estimated_map)
    axs[1].set_title('Estimated')
    axs[2].imshow(standard_deviation_map)
    axs[2].set_title('Variance')
    plt.show()
