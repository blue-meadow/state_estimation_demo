import numpy as np
from scipy.stats import multivariate_normal


def get_initial_particles(N, initial_state, transition_cov):
    return multivariate_normal.rvs(mean=initial_state,
        cov=transition_cov, size=N)

def get_particle_filter_resample(observation_likelihood_function,
                                 transition_cov):

    def resample(particles, observation, command):
        N = particles.shape[0]
        # compute the likelihood of each particle
        likelihoods = []
        for p_state in particles:
            likelihoods.append(observation_likelihood_function(
                p_state, command, observation))

        # and from the likelihoods, a distribution over the particles
        likelihoods = np.array(likelihoods)
        probs = likelihoods/likelihoods.sum()
        # sample new particles
        new_indices = np.random.choice(np.arange(N), size=N, p=probs)
        new_particles = particles[new_indices]
        # and add update noise
        new_particles += multivariate_normal.rvs(cov=transition_cov, size=N)
        return new_particles

    return resample
