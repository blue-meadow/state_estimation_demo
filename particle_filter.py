import numpy as np
from scipy.stats import multivariate_normal


def get_initial_particles(N, initial_state, transition_cov):
    return multivariate_normal.rvs(mean=initial_state,
        cov=transition_cov, size=N)

def get_particle_filter_resample(H_likelihood, dynamics, transition_cov, dt):

    def resample(particles, observation, command):
        N = particles.shape[0]
        # compute the likelihood of each particle
        likelihoods = []
        new_particles = []
        for p_state in particles:
            p_statedot = dynamics(p_state, command)
            p_state_next = p_state + p_statedot * dt
            new_particles.append(p_state_next)
            likelihoods.append(H_likelihood(p_state_next, p_statedot,
                command, observation))

        new_particles = np.array(new_particles)
        # and from the likelihoods, a distribution over the particles
        likelihoods = np.array(likelihoods)
        probs = likelihoods/likelihoods.sum()
        # resample new particles
        new_indices = np.random.choice(np.arange(N), size=N, p=probs)
        new_particles = new_particles[new_indices]
        # and add update noise
        new_particles += multivariate_normal.rvs(cov=transition_cov, size=N)
        return new_particles

    return resample
