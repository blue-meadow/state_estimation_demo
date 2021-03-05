import numpy as np


def get_simulation_step(dynamics, H_sample):

    def step(state, command, dt):
        # simple euler step state update
        statedot = dynamics(state, command)
        next_state = state + statedot * dt
        # get a corresponding observation
        # NOTE(izzy): this is a little dubious because we're using the
        # acceleration from the previous timestep
        observation = H_sample(next_state, statedot, command)
        return next_state, statedot, observation

    return step