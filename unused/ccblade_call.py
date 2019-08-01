
import numpy as np
from ccblade import *

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
    basepath = '/Users/ningrsrch/Dropbox/Projects/waked-loads/5MW_AFFiles' + os.path.sep

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
    tilt = 0.
    precone = 2.5
    yaw = 0.0
    shearExp = 0.2
    shearExp = 0.
    hubHt = 80.0
    nSector = 8

    # create CCBlade object
    aeroanalysis = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                           precone, tilt, yaw, shearExp, hubHt, nSector)


    # set conditions
    # Uinf = 10.0
    Uinf = np.ones(len(r))
    # Uinf = np.array([4., 10., 10., 10., 10., 10., 10.,
                  # 10., 10., 10., 10., 10., 10., 10.,
                  # 10., 10., 10.])
    # Uinf = np.array([4., 4., 4., 4., 4., 4., 4.,
    #             4., 4., 4., 4., 4., 4., 4.,
    #             4., 4., 10.])
    # Uinf = np.random.rand(len(r))*10.
    tsr = 7.55
    pitch = 0.0
    # Omega = Uinf*tsr/Rtip * 30.0/np.pi  # convert to RPM
    Omega = 10.
    # azimuth = 90
    azimuth = 90.

    # evaluate distributed loads
    Np, Tp = aeroanalysis.distributedAeroLoads(Uinf, Omega, pitch, azimuth)
    # print 'Np: ', Np
    # print 'Tp: ', Tp

    # plot
    import matplotlib.pyplot as plt
    # rstar = (rload - rload[0]) / (rload[-1] - rload[0])
    plt.plot(r, Tp/1e3, 'k', label='lead-lag')
    plt.plot(r, Np/1e3, 'r', label='flapwise')
    plt.xlabel('blade fraction')
    plt.ylabel('distributed aerodynamic loads (kN)')
    plt.legend(loc='upper left')

    # P, T, Q, M, CP, CT, CQ, CM = aeroanalysis.evaluate([Uinf], [Omega], [pitch], coefficients=True)

    # print(CP, CT, CQ)


    # tsr = np.linspace(2, 14, 50)
    # Omega = 10.0 * np.ones_like(tsr)
    # Uinf = Omega*np.pi/30.0 * Rtip/tsr
    # pitch = np.zeros_like(tsr)
    #
    # P, T, Q, M, CP, CT, CQ, CM = aeroanalysis.evaluate(Uinf, Omega, pitch, coefficients=True)
    #
    # plt.figure()
    # plt.plot(tsr, CP, 'k')
    # plt.xlabel('$\lambda$')
    # plt.ylabel('$c_p$')
    #
    plt.show()
