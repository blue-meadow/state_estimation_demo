# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

rcParams['font.family'] = 'monospace'

from perlin_noise import PerlinNoise


class FarmLayout(object):
  """
  Used to overlay farming equipment on environment plots.
  """
  def __init__(self):
    self.xlim = [-4, 32]
    self.ylim = [-4, 24]
    self.zlim = [-20, 0]
    self.oyster_buoys = [
      (24, 0), (24, 5), (24, 10), (24, 15), (24, 20), # Col 1
      (28, 0), (28, 5), (28, 10), (28, 15), (28, 20)  # Col 2
    ]

    self.seaweed_buoys = [
      (0, 0), (0, 20), (20, 20), (20, 0)
    ]

    # 10 lines from y=0 to y=20.
    self.seaweed_lines = []
    xlocs = np.arange(0, 21, 1)
    ylocs = [0, 20]
    for x in xlocs:
      self.seaweed_lines.append([(x, ylocs[0]), (x, ylocs[1])])


def overlay_farm_2d(fig, ax, farm_layout):
  """
  Overlays buoys and kelp lines on a figure.
  """
  for sb in farm_layout.seaweed_buoys:
    ax.add_patch(plt.Circle(sb, 0.2, color='r'))

  for ob in farm_layout.oyster_buoys:
    ax.add_patch(plt.Circle(ob, 0.3, color='r'))

  for ln in farm_layout.seaweed_lines:
    p0 = ln[0]
    p1 = ln[1]
    xdata = [p0[0], p1[0]]
    ydata = [p0[1], p1[1]]
    ax.add_line(plt.Line2D(xdata, ydata, linewidth=0.4, linestyle="solid", color="white"))

  # Box around seaweed area.
  ax.add_patch(plt.Rectangle(farm_layout.seaweed_buoys[0], 20, 20, 0.1, color='white', fill=False))


def get_slice(f, res, xlim, ylim, z=0):
  w = xlim[1] - xlim[0]
  h = ylim[1] - ylim[0]
  img = np.zeros([h*res, w*res])
  for i, x in enumerate(np.linspace(xlim[0], xlim[1], w*res)):
    for j, y in enumerate(np.linspace(ylim[0], ylim[1], h*res)):
      img[j, i] = f([x, y, z])

  return img


def get_slice_vectorized(f, res, xlim, ylim, z):
  xs, ys = np.meshgrid(np.linspace(xlim[0], xlim[1], res), np.linspace(ylim[0], ylim[1], res))
  zs = np.ones([res, res]) * z
  coords = np.stack([xs, ys, zs]).reshape([3, -1]).T
  return f(coords).reshape([res, res])


def scale_data(data, vmin=0, vmax=10):
  """
  Scale input data to the range [vmin, vmax].
  """
  dmin = np.min(data)
  dmax = np.max(data)

  data_unit = (data - dmin) / (dmax - dmin)

  return data_unit*(vmax-vmin) + vmin

# def scaled_perlin_noise(positions, data_range=[0, 10]):
#   env = PerlinNoise(octaves=0.3)  # serve as concentration map
#   observations = np.array([env(p) for p in positions])

#   # Scale the observations based on the reasonable data values.
#   mean = (data_range[1] + data_range[0]) / 2.0
#   scale = data_range[1] - data_range[0]
#   observations = scale*(observations + mean) / 2.0

#   return observations, env


def plot_factory_2d():
  farm_layout = FarmLayout()

  fig = plt.figure()
  ax = fig.gca()
  ax.set_xlim(farm_layout.xlim)
  ax.set_ylim(farm_layout.ylim)
  ax.set_xlabel("East (m)")
  ax.set_ylabel("North (m)")

  overlay_farm_2d(fig, ax, farm_layout)

  return fig, ax, farm_layout


# xs = np.linspace(farm_layout.xlim[0], np.linspace(farm_layout.xlim[1], 100))
# ys = np.linspace(farm_layout.ylim[0], np.linspace(farm_layout.ylim[1], 100))
# grid_2d = np.meshgrid(xs, ys)
# positions_2d = np.vstack(map(np.ravel, grid_2d))

def plot_dissox():
  fig, ax, farm_layout = plot_factory_2d()
  depth = 10.0
  perlin1 = PerlinNoise(octaves=0.1)
  dissox = get_slice(lambda x: perlin1(x), 5, farm_layout.xlim, farm_layout.ylim, -depth)
  dissox = scale_data(dissox, vmin=6.7, vmax=8.1)
  im = ax.imshow(dissox, cmap="viridis", origin="lower",
      extent=(farm_layout.xlim[0], farm_layout.xlim[1], farm_layout.ylim[0], farm_layout.ylim[1]))
  ax.set_title("Dissolved Oxygen (mg/L) @ {:.1f}m".format(depth))
  plt.colorbar(im)
  plt.tight_layout()
  plt.show()

def plot_nutrients():
  fig, ax, farm_layout = plot_factory_2d()
  depth = 10.0
  perlin1 = PerlinNoise(octaves=0.05)
  data = get_slice(lambda x: perlin1(x), 5, farm_layout.xlim, farm_layout.ylim, -depth)
  data = scale_data(data, vmin=0.3, vmax=2.3)
  im = ax.imshow(data, cmap="magma", origin="lower",
      extent=(farm_layout.xlim[0], farm_layout.xlim[1], farm_layout.ylim[0], farm_layout.ylim[1]))
  ax.set_title("Nitrogen Content (mg/L) @ {:.1f}m".format(depth))
  plt.colorbar(im)
  plt.tight_layout()
  plt.show()


# https://stackoverflow.com/questions/30464117/plotting-a-imshow-image-in-3d-in-matplotlib
def plot_current_speed_slices():
  xx, yy = np.meshgrid(np.linspace(-4, 32, 36), np.linspace(-4, 24, 28))

  # create vertices for a rotated mesh (3D rotation matrix)
  X =  xx
  Y =  yy

  perlin1 = PerlinNoise(octaves=0.1)
  data = get_slice(lambda x: perlin1(x), 1, [-4, 32], [-4, 24], -10)

  data1 = 0.41*data + np.cos(0.053*xx + np.sin(0.1*yy))
  data2 = 0.4*data + np.cos(0.05*xx + np.sin(0.1*yy))
  data3 = 0.6*data + np.sin(0.15*xx + np.sin(0.1*yy))

  # create the figure
  fig = plt.figure()

  # show the 3D rotated projection
  ax = fig.gca(projection='3d')
  ax.plot_surface(X, Y, -5*np.ones(X.shape), rstride=1, cstride=1, facecolors=plt.cm.cool(data), shade=False)
  ax.plot_surface(X, Y, -10*np.ones(X.shape), rstride=1, cstride=1, facecolors=plt.cm.cool(data2), shade=False)
  ax.plot_surface(X, Y, -15*np.ones(X.shape), rstride=1, cstride=1, facecolors=plt.cm.cool(data3), shade=False)

  ax.set_title('Flow Speed (m/s)')
  ax.set_xlabel('East (m)')
  ax.set_ylabel('North (m)')
  ax.set_zlabel('Depth (m)')
  plt.colorbar(plt.cm.ScalarMappable(cmap="cool"), ax=ax)

  plt.show()


def plot_currents_3d():
  farm_layout = FarmLayout()

  fig = plt.figure()
  ax = fig.gca(projection='3d')

  # Make the grid
  x, y, z = np.meshgrid(np.arange(farm_layout.xlim[0], farm_layout.xlim[1], 3.0),
                        np.arange(farm_layout.ylim[0], farm_layout.ylim[1], 3.0),
                        np.arange(farm_layout.zlim[0], farm_layout.zlim[1], 5.0))

  # Make the direction data for the arrows
  u = 100.0 * np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
  v = 100.0 * -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
  # w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
  #    np.sin(np.pi * z))

  w = z*0

  ax.quiver(x, y, z, u, v, w, length=2.0, normalize=True)

  plt.show()


def plot_depth_charts():
  fig, ax = plt.subplots()
  fig.subplots_adjust(right=0.75)

  ax2 = ax.twinx()
  ax3 = ax.twinx()
  ax2.set_ylabel("Temperature (deg C)")
  # ax3.set_ylabel("Illumination (PAR)")

  ds = np.linspace(0.0, 20.0, 30)

  dissox = -0.793 * np.log(ds) + 2.2956 + 0.1*np.random.random(size=len(ds))
  CO2 = 0.12*ds + 0.001*(ds-5)**3 + 0.1 + 0.15*np.random.random(size=len(ds))
  light = 100.0 / (ds + 1.0)
  temp = 19.0 - (0.15*(ds - 10))**3 - 0.1*ds + 0.05*np.random.random(size=len(ds))

  p1, = ax.plot(ds, dissox, "rx", label="Dissolved Oxygen")
  p2, = ax.plot(ds, CO2, "gx", label="Dissolved CO2")
  p3, = ax2.plot(ds, temp, "bx", label="Ocean Temperature")
  # p4, = ax3.plot(ds, light, "yx", label="Light")
  plt.title("Depth Plots")
  plt.xlabel("Depth (m)")
  ax.set_ylabel("Concentration (mg/L)")
  ax.legend(handles=[p1, p2, p3], loc='upper right')
  plt.show()


def plot_growth():
  fig = plt.figure()
  ax = fig.gca()

  days = np.arange(0, 60)

  growth = 40.0 + 0.6*days + 3.6*np.sin(days*0.1 - 10) + 3.9*np.random.random(size=len(days))

  ax.plot(days, growth, "gx")
  ax.set_xlabel("day")
  ax.set_ylabel("average frond length (cm)")
  ax.set_title("Kelp Growth (last 60 days)")
  plt.show()

# plot_dissox()
# plot_nutrients()

# plot_currents_3d()
# plot_current_speed_slices()
# plot_depth_charts()
plot_growth()
