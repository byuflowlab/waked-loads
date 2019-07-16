
import numpy as np
from ccblade import *
from wakeLoadsFuncs import *
import matplotlib.pyplot as plt

def a_of_t(t,rpm):
    rps = rpm/60.
    a = ((rps*t)%1)*360.
    return a


if __name__ == '__main__':

    turbineX = np.array([0.,505.6])
    turbineY = np.array([0.,0.])
    turbineZ = np.array([90.,90.])

    num = 1000
    y = np.linspace(-200.,200.,num)
    z = np.linspace(0.,200.,num)

    y_locs,z_locs = np.meshgrid(y,z)
    y_locs = np.ndarray.flatten(y_locs)
    z_locs = np.ndarray.flatten(z_locs)

    x_locs = np.ones_like(y_locs)*505.6

    speeds = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, 8.,shearExp=0.)

    Z = np.zeros((num,num))
    Y = np.zeros((num,num))
    S = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            Z[i][j] = z_locs[i*num+j]
            Y[i][j] = y_locs[i*num+j]
            S[i][j] = speeds[i*num+j]
    plt.pcolormesh(Y,Z,S)
    c = plt.Circle((0.,90.),47.118806929895634, color='b', fill=False)
    plt.gca().add_artist(c)
    c = plt.Circle((0.,90.),2.*47.118806929895634, color='b', fill=False)
    plt.gca().add_artist(c)
    plt.axis('equal')
    plt.colorbar()
    plt.show()
