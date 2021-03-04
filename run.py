import numpy as np

from simulate import simple_2d_step, simple_2d_get_initial
from visualize import simple_2d_visualize_logs


def simple_2d_run():
    T = 1000   # number of simulation steps
    dt = 0.05 # timestep
    time_log = np.arange(T) * dt
    commands = np.ones([T-1, 2]) * np.array([[0.5, 2.0]])

    state, particles = simple_2d_get_initial()

    state_log = [state]
    particle_log = [particles]
    for command in commands:
        state, particles = simple_2d_step(state, command, particles, dt)
        state_log.append(state)
        particle_log.append(particles)

    return time_log, np.array(state_log), np.array(particle_log)


if __name__ == '__main__':
    logs = simple_2d_run()
    simple_2d_visualize_logs(logs)