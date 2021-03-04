import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def simple_2d_animate_step(i, *logs):
    # set up the plotting window
    plt.clf()
    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 4.5)
    plt.gca().set_aspect('equal', adjustable='box')
    # get this timestep from the logs
    t, s, p = (l[i] for l in logs)
    # plot the particles
    plt.scatter(*p[:,:2].T, c='b', s=1, alpha=0.1)
    # and the true state
    plt.scatter(s[0], s[1], c='r')
    plt.title(f'Time = {t:.3f} seconds')

def simple_2d_visualize_logs(logs):
    T = logs[0].shape[0]
    # run the animation
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, simple_2d_animate_step, T, interval=50, fargs=logs)
    plt.show()
