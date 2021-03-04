import numpy as np

from dynamics import simple_2d_dynamics, simple_2d_transition_cov
from observation import simple_2d_H_sample, \
    simple_2d_H_likelihood
from particle_filter import get_particle_filter_step, get_initial_particles


def simple_2d_get_initial():
    state = np.zeros(6)
    particles = get_initial_particles(100, state, simple_2d_transition_cov)
    return state, particles

def simple_2d_step(state, command, particles, dt):
    # ----------------- this performs the simulation -----------------

    # simple euler step state update
    statedot = simple_2d_dynamics(state, command)
    next_state = state + statedot * dt
    # get a corresponding observation
    # NOTE(izzy): this is a little dubious because we're using the
    # acceleration from the previous timestep
    observation = simple_2d_H_sample(next_state, statedot, command)

    # ----------------- and this update state estimator -----------------

    # get function to step particles
    step = get_particle_filter_step(simple_2d_H_likelihood,
        simple_2d_dynamics, simple_2d_transition_cov, dt)

    # step the particles based on the observation
    new_particles = step(particles, observation, command)

    return next_state, new_particles