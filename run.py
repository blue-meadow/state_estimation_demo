import numpy as np

from dynamics import simple_2d_dynamics, simple_2d_transition_cov
from observation import simple_2d_H_sample, simple_2d_H_likelihood, \
    simple_2d_observation_cov
from particle_filter import get_particle_filter_step
from simulate import get_simulation_step
from visualize import simple_2d_visualize_logs

def simple_2d_run():
    T = 1000   # number of simulation steps
    dt = 0.05 # timestep

    # get step functions for dynamics and particle filter
    simulation_step = get_simulation_step(simple_2d_dynamics,
                                          simple_2d_H_sample)
    particle_filter_step = get_particle_filter_step(simple_2d_dynamics,
                                                    simple_2d_H_likelihood,
                                                    simple_2d_transition_cov)

    # get initial conditions and trajectory of commands
    state = np.zeros(6)
    particles = np.random.randn(1000, 6)
    commands = np.ones([T-1, 2]) * np.array([[0.5, 2.0]])

    # create logs
    time_log = np.arange(T) * dt
    state_log = [state]
    particle_log = [particles]

    for command in commands:
        # step
        state, statedot, observation = simulation_step(state, command, dt)
        particles = particle_filter_step(particles, observation, command, dt)
        # log
        state_log.append(state)
        particle_log.append(particles)

    return time_log, np.array(state_log), np.array(particle_log)


if __name__ == '__main__':
    logs = simple_2d_run()
    simple_2d_visualize_logs(logs)
