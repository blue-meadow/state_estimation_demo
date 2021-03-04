import numpy as np
from matplotlib import pyplot as plt

from simulate import simple_2d_step, simple_2d_get_initial


def simple_2d_run():
    commands = np.ones([49, 2]) * np.array([[0.0, 1.0]])
    dt = 0.05
    state, particles = simple_2d_get_initial()

    state_log = [state]
    particle_log = [particles]
    for command in commands:
        state, particles = simple_2d_step(state, command, particles, dt)
        print(particles)
        state_log.append(state)
        particle_log.append(particles)

    return np.array(state_log), np.array(particle_log)


if __name__ == '__main__':
    s_log, p_log = simple_2d_run()
    for s, p in zip(s_log, p_log):
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.scatter(*p[:,:2].T, c='b')
        plt.scatter(s[0], s[1], c='r')
        plt.show()
