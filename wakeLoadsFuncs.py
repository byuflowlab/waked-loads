
import numpy as np
from ccblade import *
from scipy import interpolate
# from _porteagel_fortran import porteagel_analyze as porteagel_analyze_fortran
# from _porteagel_fortran import porteagel_visualize
# import _porteagel_fortran
import gaus
from wakeexchange.utilities import sunflower_points
import sys
import time
sys.dont_write_bytecode = True

def calc_moment_edge(Uinf,loc,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=90.,shearExp=0.):

    N = len(r)
    M = 0.

    yaw = 0.
    tilt = 0.

    # s = time.time()
    aeroanalysis = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                           precone, tilt, yaw, shearExp, hubHt, nSector)
    _, loads_edge = aeroanalysis.distributedAeroLoads(Uinf, Omega, pitch, azimuth)
    # print 'run CCBlade: ', time.time()-s

    # s = time.time()

    # #approximate load at r = Rhub
    # dL = loads_flap[1]-loads_flap[0]
    # dr = r[1]-r[0]
    # m = dL/dr
    # Lhub = np.array([loads_flap[0] + m*(Rhub-r[0])])
    #
    # #approximate load at r = Rtip
    # dL = loads_flap[-1]-loads_flap[-2]
    # dr = r[-1]-r[-2]
    # m = dL/dr
    # Ltip = np.array([loads_flap[-1] + m*(Rtip-r[-1])])
    #
    # Rhub = np.array([Rhub])
    # Rtip = np.array([Rtip])
    #
    r = np.append(Rhub,r)
    r = np.append(r,Rtip)
    # loads_flap = np.append(Lhub,loads_flap)
    # loads_flap = np.append(loads_flap,Ltip)
    #
    # L = interpolate.interp1d(r,loads_flap)
    # rad = np.linspace(Rhub,Rtip,500)
    # x = L(rad)*(rad-loc)
    # M_flap = np.trapz(x,x=rad)



    #approximate load at r = Rhub
    dL = loads_edge[1]-loads_edge[0]
    dr = r[1]-r[0]
    m = dL/dr
    Lhub = np.array([loads_edge[0] + m*(Rhub-r[0])])

    #approximate load at r = Rtip
    dL = loads_edge[-1]-loads_edge[-2]
    dr = r[-1]-r[-2]
    m = dL/dr
    Ltip = np.array([loads_edge[-1] + m*(Rtip-r[-1])])

    loads_edge = np.append(Lhub,loads_edge)
    loads_edge = np.append(loads_edge,Ltip)

    L = interpolate.interp1d(r,loads_edge)
    rad = np.linspace(Rhub,Rtip,500)
    x = L(rad)*(rad-loc)
    M_edge = np.trapz(x,x=rad)

    """add gravity loads"""
    blade_mass = 17536.617
    blade_cm = 20.650
    grav = 9.81
    M_edge += np.sin(np.deg2rad(azimuth))*blade_mass*grav*blade_cm

    # print 'the rest: ', time.time()-s
    return M_edge


def calc_moment(Uinf,loc,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=90.,shearExp=0.):

    N = len(r)
    M = 0.

    yaw = 0.
    tilt = 0.

    s = time.time()
    aeroanalysis = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                           precone, tilt, yaw, shearExp, hubHt, nSector)
    loads_flap, loads_edge = aeroanalysis.distributedAeroLoads(Uinf, Omega, pitch, azimuth)
    print 'loads_edge: ', loads_edge
    # print 'run CCBlade: ', time.time()-s

    s = time.time()

    #approximate load at r = Rhub
    dL = loads_flap[1]-loads_flap[0]
    dr = r[1]-r[0]
    m = dL/dr
    Lhub = np.array([loads_flap[0] + m*(Rhub-r[0])])

    #approximate load at r = Rtip
    dL = loads_flap[-1]-loads_flap[-2]
    dr = r[-1]-r[-2]
    m = dL/dr
    Ltip = np.array([loads_flap[-1] + m*(Rtip-r[-1])])

    Rhub = np.array([Rhub])
    Rtip = np.array([Rtip])

    r = np.append(Rhub,r)
    r = np.append(r,Rtip)
    loads_flap = np.append(Lhub,loads_flap)
    loads_flap = np.append(loads_flap,Ltip)

    L = interpolate.interp1d(r,loads_flap)
    rad = np.linspace(Rhub,Rtip,500)
    x = L(rad)*(rad-loc)
    M_flap = np.trapz(x,x=rad)



    #approximate load at r = Rhub
    dL = loads_edge[1]-loads_edge[0]
    dr = r[1]-r[0]
    m = dL/dr
    Lhub = np.array([loads_edge[0] + m*(Rhub-r[0])])

    #approximate load at r = Rtip
    dL = loads_edge[-1]-loads_edge[-2]
    dr = r[-1]-r[-2]
    m = dL/dr
    Ltip = np.array([loads_edge[-1] + m*(Rtip-r[-1])])

    loads_edge = np.append(Lhub,loads_edge)
    loads_edge = np.append(loads_edge,Ltip)

    L = interpolate.interp1d(r,loads_edge)
    rad = np.linspace(Rhub,Rtip,500)
    x = L(rad)*(rad-loc)
    M_edge = np.trapz(x,x=rad)

    """add gravity loads"""
    blade_mass = 17536.617
    blade_cm = 20.650
    grav = 9.81
    M_edge += np.sin(np.deg2rad(azimuth))*blade_mass*grav*blade_cm

    # print 'the rest: ', time.time()-s
    return M_flap,M_edge


def findXY(x_hub,y_hub,r,yaw_deg):
    # assuming the x and y coordinates have been rotated such that the wind is
    # coming from left to right
    # yaw CCW positive
    sy = np.sin(np.deg2rad(yaw_deg))
    cy = np.cos(np.deg2rad(yaw_deg))

    sideN_x = x_hub - r*sy
    sideN_y = y_hub + r*cy
    sideS_x = x_hub + r*sy
    sideS_y = y_hub - r*cy

    return sideN_x,sideN_y,sideS_x,sideS_y


def findXYZ(x_hub,y_hub,z_hub,r,yaw_deg,azimuth_deg):
    # assuming the x and y coordinates have been rotated such that the wind is
    # coming from left to right
    # yaw CCW positive
    sy = np.sin(np.deg2rad(yaw_deg))
    cy = np.cos(np.deg2rad(yaw_deg))

    sa = np.sin(np.deg2rad(azimuth_deg))
    ca = np.cos(np.deg2rad(azimuth_deg))

    x_locs = x_hub - r*sy*sa
    y_locs = y_hub - r*cy*sa
    z_locs = z_hub + r*ca


    return x_locs, y_locs, z_locs


def get_speeds(turbineX, turbineY, xpoints, ypoints, zpoints, wind_speed, wec_factor=1.0,wake_model_version=2016,sm_smoothing=700.,
                calc_k_star=True,ti_calculation_method=2,wake_combination_method=1,shearExp=0.,TI=0.11):
    wec_factor = 1.
    nTurbines = len(turbineX)
    turbineZ = np.ones(nTurbines)*90.
    yaw = np.zeros(nTurbines)
    rotorDiameter = np.ones(nTurbines)*126.4
    ky = 0.022
    kz = 0.022
    alpha = 2.32
    beta = 0.154
    I = TI
    # I = 0.056
    z_ref = 50.
    z_0 = 0.
    shear_exp = shearExp
    # RotorPointsY = np.array([0.])
    # RotorPointsZ = np.array([0.])
    nRotorPoints = 20
    RotorPointsY, RotorPointsZ = sunflower_points(nRotorPoints)
    velX = xpoints
    velY = ypoints
    velZ = zpoints

    sorted_x_idx = np.argsort(turbineX)

    use_ct_curve = False
    interp_type = 1.
    Ct = np.ones(nTurbines)*8./9.
    ct_curve_wind_speed = np.ones_like(Ct)*wind_speed
    ct_curve_ct = Ct

    ws_array, wake_diameters = gaus.porteagel_visualize(turbineX, sorted_x_idx, turbineY, turbineZ, rotorDiameter, Ct,
                                           wind_speed, yaw, ky, kz, alpha, beta, I, RotorPointsY,
                                           RotorPointsZ, z_ref, z_0, shear_exp, velX, velY, velZ,
                                           wake_combination_method, ti_calculation_method, calc_k_star,
                                           wec_factor, wake_model_version, interp_type, use_ct_curve,
                                           ct_curve_wind_speed, ct_curve_ct, sm_smoothing)

    return ws_array, wake_diameters


def get_eff_turbine_speeds(turbineX, turbineY, wind_speed, wec_factor=1.0,wake_model_version=2016,sm_smoothing=700.,
                calc_k_star=True,ti_calculation_method=2,wake_combination_method=1,shearExp=0.,TI=0.11):
    wec_factor = 1.
    nTurbines = len(turbineX)
    turbineZ = np.ones(nTurbines)*90.
    yaw = np.zeros(nTurbines)
    rotorDiameter = np.ones(nTurbines)*126.4
    ky = 0.022
    kz = 0.022
    alpha = 2.32
    beta = 0.154
    I = TI
    # I = 0.056
    z_ref = 50.
    z_0 = 0.
    shear_exp = shearExp
    # RotorPointsY = np.array([0.])
    # RotorPointsZ = np.array([0.])
    nRotorPoints = 20
    RotorPointsY, RotorPointsZ = sunflower_points(nRotorPoints)

    sorted_x_idx = np.argsort(turbineX)

    use_ct_curve = False
    interp_type = 1.
    Ct = np.ones(nTurbines)*8./9.
    ct_curve_wind_speed = np.ones_like(Ct)*wind_speed
    ct_curve_ct = Ct
    print_ti = False

    ws_array = gaus.porteagel_analyze(turbineX,sorted_x_idx,turbineY,
                        turbineZ,rotorDiameter,Ct,wind_speed,yaw,ky,kz,alpha,beta,
                        I,RotorPointsY,RotorPointsZ,z_ref,z_0,shear_exp,wake_combination_method,
                        ti_calculation_method,calc_k_star,wec_factor,print_ti,wake_model_version,
                        interp_type,use_ct_curve,ct_curve_wind_speed,ct_curve_ct,sm_smoothing)

    return ws_array

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

    # locs = np.linspace(Rhub,Rtip,100)
    # M = np.zeros(100)
    # for i in range(100):
    #     M[i] = calc_moment(Uinf,locs[i],r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,
    #                                     hubHt,nSector,Omega,pitch)
    # print M

    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(15,5))
    # ax1 = plt.subplot(131)
    # ax2 = plt.subplot(132)
    # ax3 = plt.subplot(133)
    #
    #
    # az = np.linspace(0.,720.*5,300)
    # for i in range(100):
    #     ax1.cla()
    #     ax2.cla()
    #     ax3.cla()
    #     ax1.axis('off')
    #     ax2.axis('off')
    #     ax3.axis('off')
    #     ax1.axis('square')
    #     ax2.axis('square')
    #     ax3.axis('square')
    #     ax1.set_ylim(-70.,70.)
    #     ax2.set_ylim(30.,170.)
    #     ax3.set_ylim(30.,170.)
    #     ax1.set_xlim(-10.,10.)
    #     ax2.set_xlim(-10.,10.)
    #     ax3.set_xlim(70.,-70.)
    #
    #     azimuth_deg = az[i]
    #     yaw_deg = 0.
    #     x_locs,y_locs,z_locs = findXYZ(0.,0.,100.,r,yaw_deg,azimuth_deg)
    #
    #
    #     if az[i]%360 < 90. or az[i]%360 > 270.:
    #         ax1.plot(0.,0.,'ob',markersize=15)
    #     if az[i]%360 < 180.:
    #         ax2.plot(0.,100.,'ob',markersize=15)
    #     ax3.plot(0.,100.,'ob',markersize=15)
    #
    #     ax1.plot(x_locs,y_locs,'or')
    #     ax2.plot(x_locs,z_locs,'or')
    #     ax3.plot(y_locs,z_locs,'or')
    #
    #
    #     if az[i]%360 > 90. and az[i]%360 < 270.:
    #         ax1.plot(0.,0.,'ob',markersize=15)
    #     if az[i]%360 > 180.:
    #         ax2.plot(0.,100.,'ob',markersize=15)
    #
    #
    #     plt.pause(0.01)

    turbineX = np.array([0.,504.])
    turbineY = np.array([0.,126.])
    wind_speed = 8.

    x1,y1,x2,y2 = findXY(turbineX[0],turbineY[0],r,0.)
    speeds1 = get_speeds(turbineX, turbineY, x1, y1, wind_speed)
    speeds2 = get_speeds(turbineX, turbineY, x2, y2, wind_speed)
    M1_flap, M1_edge = calc_moment(speeds1,0.,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch)
    M2_flap, M2_edge = calc_moment(speeds2,0.,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch)

    print M1_flap-M2_flap
    print M1_edge-M2_edge

    x1,y1,x2,y2 = findXY(turbineX[1],turbineY[1],r,0.)
    speeds1 = get_speeds(turbineX, turbineY, x1, y1, wind_speed)
    speeds2 = get_speeds(turbineX, turbineY, x2, y2, wind_speed)
    M1_flap, M1_edge = calc_moment(speeds1,0.,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch)
    M2_flap, M2_edge = calc_moment(speeds2,0.,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch)

    print M1_flap-M2_flap
    print M1_edge-M2_edge


    # import matplotlib.pyplot as plt
    # plt.plot(x1,y1,'o',color='C0')
    # plt.plot(x2,y2,'o',color='C0')
    # plt.plot(0.,0.,'o',color='C1')
    # plt.axis('equal')
    # plt.show()

    # plt.plot(locs,M)
    # plt.show()

    # plt.plot(r, Tp/1e3, 'k', label='lead-lag')
    # plt.plot(r, Np/1e3, 'r', label='flapwise')
    # plt.xlabel('blade fraction')
    # plt.ylabel('distributed aerodynamic loads (kN)')
    # plt.legend(loc='upper left')
    # plt.show()
