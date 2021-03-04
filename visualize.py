import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def simple_2d_animate_step(i, *logs):
    # set up the plotting window
    plt.clf()
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    # get this timestep from the logs
    s, p = (l[i] for l in logs)
    # plot the particles
    plt.scatter(*p[:,:2].T, c='b', s=1, alpha=0.1)
    # and the true state
    plt.scatter(s[0], s[1], c='r')

def simple_2d_visualize_logs(logs):
    # run the animation
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, simple_2d_animate_step, 200, interval=50, fargs=logs)
    plt.show()
