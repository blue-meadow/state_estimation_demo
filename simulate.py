import numpy as np

from dynamics import simple_2d_dynamics, simple_2d_transition_cov
from observation import simple_2d_acc_gyro_mag_sample, \
    simple_2d_acc_gyro_mag_likelihood
from particle_filter import get_particle_filter_resample, get_initial_particles


def simple_2d_get_initial():
    state = np.zeros(6)
    particles = get_initial_particles(100, state, simple_2d_transition_cov)
    return state, particles

def simple_2d_step(state, command, particles, dt):
    # ----------------- this performs the simulation -----------------

    # simple euler step state update
    next_state = state + simple_2d_dynamics(state, command) * dt
    # get a corresponding observation
    # NOTE(izzy): this is a little dubious because we're using the 
    # same command on the next timestep to compute acceleration
    observation = simple_2d_acc_gyro_mag_sample(next_state, command)

    # ----------------- and this update state estimator -----------------

    # get function to resample particles
    resample = get_particle_filter_resample(
        simple_2d_acc_gyro_mag_likelihood, simple_2d_transition_cov)

    # resample the particles based on the observation
    new_particles = resample(particles, observation, command)

    return next_state, new_particles