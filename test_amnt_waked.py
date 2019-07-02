
import numpy as np
from ccblade import *
from wakeLoadsFuncs import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as TIME



def amount_waked(dy,wake_radius,rotor_diameter,az):

    r = rotor_diameter/2.

    if az%360. == 0. or az%360. == 180.:
        m = 1000.
    else:
        m = np.cos(np.deg2rad(az))/np.sin(np.deg2rad(az))
    b = -m*dy

    p = np.array([m**2+1.,2.*m*b,b**2-wake_radius**2])
    x = np.roots(p)

    # they don't intersect
    if np.imag(x[0]) != 0. or np.imag(x[1]) != 0.:
        amnt_waked = 0.
        dist = False

    # they do intersect
    else:
        y = m*x+b
        dir_blade = np.array([r*np.cos(np.deg2rad(90.-az%360.)),r*np.sin(np.deg2rad(90.-az%360.))])
        dir_intersect1 = np.array([x[0]-dy,y[0]])
        dir_intersect2 = np.array([x[1]-dy,y[1]])

        d1 = np.dot(dir_blade,dir_intersect1)
        d2 = np.dot(dir_blade,dir_intersect2)

        if d1 < 0. and d2 < 0.:
            amnt_waked = 0.
            dist = False
        else:
            if d1 < 0. and d2 > 0.:
                dist = np.sqrt((x[1]-dy)**2+(y[1])**2)
            elif d1 > 0. and d2 < 0.:
                dist = np.sqrt((x[0]-dy)**2+(y[0])**2)
            else:
                if d1 <= d2:
                    dist = np.sqrt((x[0]-dy)**2+(y[0])**2)
                elif d2 < d1:
                    dist = np.sqrt((x[1]-dy)**2+(y[1])**2)

            if abs(dy) > wake_radius:
                if dist > r:
                    amnt_waked = 0.
                else:
                    amnt_waked = 1.-dist/r
            else:
                if dist > r:
                    amnt_waked = 1.
                else:
                    amnt_waked = dist/r




    # if amnt_waked > 1.:
    #     amnt_waked = 1.
    print 'dist: ', dist
    print 'amnt_waked: ', amnt_waked

    plt.cla()
    plt.plot(0.,0.,'ob')
    plt.plot(dy,0.,'or')
    wake = plt.Circle((0.,0.),wake_radius,color='blue',fill=False)
    rotor = plt.Circle((dy,0.),r,color='red',fill=False)
    plt.gca().add_patch(wake)
    plt.gca().add_patch(rotor)
    plt.plot(np.array([dy,dy+r*np.cos(np.deg2rad(90.-az%360.))]),np.array([0.,r*np.sin(np.deg2rad(90.-az%360.))]),'r')

    plt.axis('equal')
    plt.xlim(-400.,400.)
    plt.pause(2.)


if __name__ == '__main__':

    rotor_diameter = 100.
    wake_radius = 60.
    dy = 40.

    print 'dy: ', dy
    print 'wake_radius: ', wake_radius
    print 'rotor_diameter: ', rotor_diameter

    az = np.linspace(0.,360.,15)
    # az = np.array([290.])

    for i in range(len(az)):
        amount_waked(dy,wake_radius,rotor_diameter,az[i])

    plt.show()
