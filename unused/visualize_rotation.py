
import numpy as np
from ccblade import *
from wakeLoadsFuncs import *

if __name__ == '__main__':

    # geometry
    Rhub = 1.5
    Rtip = 63.0

    r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                  28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                  56.1667, 58.9000, 61.6333])
    chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                      3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
    theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                      6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
    B = 3  # number of blades

    # atmosphere
    rho = 1.225
    mu = 1.81206e-5

    import os
    afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
    basepath = '5MW_AFFiles' + os.path.sep

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
    airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
    airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
    airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
    airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
    airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
    airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
    airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    af = [0]*len(r)
    for i in range(len(r)):
        af[i] = airfoil_types[af_idx[i]]


    tilt = -5.0
    precone = 2.5
    shearExp = 0.2
    hubHt = 80.0
    nSector = 8

    Uinf = np.array([4., 4., 10., 4., 4., 4., 10.,
                  10., 10., 10., 10., 10., 10., 10.,
                  10., 10., 10.])

    tsr = 7.55
    pitch = 0.0
    Omega = 10.


    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)


    az = np.linspace(0.,720.*5,300)
    for i in range(100):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.axis('square')
        ax2.axis('square')
        ax3.axis('square')
        ax1.set_ylim(-70.,70.)
        ax2.set_ylim(0.,170.)
        ax3.set_ylim(0.,170.)
        ax1.set_xlim(-50.,50.)
        ax2.set_xlim(-50.,50.)
        ax3.set_xlim(70.,-70.)

        azimuth_deg = az[i]
        yaw_deg = 0.
        x_locs,y_locs,z_locs = findXYZ(0.,0.,100.,r,yaw_deg,azimuth_deg)

        ax1.plot(5.,0.,'ok',markersize=20)
        ax2.plot(5.,100.,'ok',markersize=20)
        ax2.plot(np.array([3.,0.,10.,7.]),np.array([100.,0.,0.,100.]),'-k',linewidth=2)
        ax3.plot(0.,100.,'ok',markersize=25)
        ax3.plot(np.array([-3.,-5.,5.,3.]),np.array([100.,0.,0.,100.]),'-k',linewidth=2)
        if az[i]%360 < 90. or az[i]%360 > 270.:
            ax1.plot(0.,0.,'ob',markersize=15)
        if az[i]%360 < 180.:
            ax2.plot(0.,100.,'ob',markersize=15)
        ax3.plot(0.,100.,'ob',markersize=15)

        ax1.plot(x_locs,y_locs,'or')
        ax2.plot(x_locs,z_locs,'or')
        ax3.plot(y_locs,z_locs,'or')


        if az[i]%360 > 90. and az[i]%360 < 270.:
            ax1.plot(0.,0.,'ob',markersize=15)
        if az[i]%360 > 180.:
            ax2.plot(0.,100.,'ob',markersize=15)



        plt.pause(0.01)
