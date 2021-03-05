import numpy as np
from scipy.stats import multivariate_normal


def get_initial_particles(N, initial_state, transition_cov):
    return multivariate_normal.rvs(mean=initial_state,
        cov=transition_cov, size=N)

def get_particle_filter_step(dynamics, H_likelihood, transition_cov):

    def step(particles, observation, command, dt):
        N = particles.shape[0]

        # simulate the particles forward according to the command
        particlesdot = dynamics(particles, command)
        new_particles = particles + particlesdot * dt

        # compute the likelihood of each particle
        likelihoods = H_likelihood(new_particles, particlesdot,
                command, observation)

        # and from the likelihoods, a distribution over the particles
        probs = likelihoods/likelihoods.sum()
        # resample new particles
        new_indices = np.random.choice(np.arange(N), size=N, p=probs)
        new_particles = new_particles[new_indices]
        # and add update
        # NOTE(izzy) here, noise is proportional to simulation step size
        # if the vehicle moves farther, we are less certain of its location
        new_particles += multivariate_normal.rvs(cov=transition_cov*dt, size=N)
        return new_particles

    return step
