
import numpy as np
from ccblade import *
from wakeLoadsFuncs import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as TIME

def a_of_t(t,rpm):
    rps = rpm/60.
    a = ((rps*t)%1)*360.
    return a


# def M_of_a_func(M,a):



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
    tilt = 0.
    precone = 0.
    shearExp = 0.
    hubHt = 90.0
    nSector = 1

    tsr = 7.55
    pitch = 0.0
    Omega = 7.506


    mult = np.linspace(0.,1.5,4)
    mult = np.array([0.])

    turbineX = np.array([0.,505.6])
    # turbineY = np.array([0.,126.4])*1.5
    wind_speed = 10.

    yaw_deg = 0.

    angles = np.linspace(0.,360.,100)
    mom_flap1 = np.zeros_like(angles)
    mom_edge1 = np.zeros_like(angles)

    mom_flap2 = np.zeros_like(angles)
    mom_edge2 = np.zeros_like(angles)

    mm = 1.0
    turbineY1 = np.array([0.,0.])
    turbineY2 = np.array([0.,1.])*55.3235576*mm
    for i in range(len(angles)):
        az = angles[i]
        x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY1[1],90.,r,yaw_deg,az)
        speeds, _ = get_speeds(turbineX, turbineY1, x_locs, y_locs, z_locs, wind_speed,shearExp=shearExp)
        mom_flap1[i], mom_edge1[i] = calc_moment(speeds,r[0],r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az,shearExp=shearExp)

        x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY2[1],90.,r,yaw_deg,az)
        speeds, diameters = get_speeds(turbineX, turbineY2, x_locs, y_locs, z_locs, wind_speed,shearExp=shearExp)
        mom_flap2[i], mom_edge2[i] = calc_moment(speeds,r[0],r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az,shearExp=shearExp)

    f_flap1 = interp1d(angles, mom_flap1, kind='cubic')
    f_flap2 = interp1d(angles, mom_flap2, kind='cubic')

    f = plt.figure(figsize=(10,5))
    ax1 = plt.subplot(111)

    # ax2 = plt.subplot(122)


    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C720_W8_T11.0_P0.5_m2D_L1.0/Model.out'
    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C433_W8_T5.6_P0.0_4D_L-1.0/Model.out'
    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C649_W8_T11.0_P0.0_4D_L-1.0/Model.out'
    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C437_W8_T5.6_P0.0_4D_L0/Model.out'
    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C475_W8_T5.6_P0.5_4D_L0.5/Model.out'
    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
    lines = np.loadtxt(filename,skiprows=8)
    time = lines[:,0]
    mx = lines[:,11]
    my = lines[:,12]
    om = lines[:,6]
    num = 2000
    t = time[0:num]
    o = om[0:num]
    my_fast = my[0:num]
    mx_fast = mx[0:num]
    M_flap = np.zeros_like(my_fast)
    M_edge = np.zeros_like(mx_fast)

    t = t-t[0]


    current_pos = 0.
    s = np.array([])
    for j in range(len(mult)):
        print j
        # turbineY = np.array([0.,126.4])*mult[j]
        # turbineY = np.array([0.,126.4])*0.
        # turbineY = np.array([0.,1000000000.])
        for i in range(len(t)-1):
            # az = a_of_t(t[i],Omega)
            current_pos = (current_pos+((o[i+1]+o[i])/2.*(t[i+1]-t[i])/60.)*360.)%360.
            az = current_pos
            # az = (90.*float(i))%360.
            # print 'az: ', az
            # x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY[1],90.,r,yaw_deg,az)
            # speeds = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, wind_speed,shearExp=shearExp)
            # s = np.append(s,speeds[0])
            # # M_flap[i], M_edge[i] = calc_moment(speeds,r[0],r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az,shearExp=shearExp)
            # M_flap[i], M_edge[i] = calc_moment(speeds,r[0],r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,o[i],pitch,azimuth=az,shearExp=shearExp)
            M_flap[i] = f_flap1(az)

        # ax1.plot(np.deg2rad(azimuth_deg),M_flap/1000.)#,label='%sD'%mult[j])
        # ax2.plot(np.deg2rad(azimuth_deg),M_edge/1000.)
        M_flap = M_flap/1000.
        # M_edge = M_edge/1000.
        # ax1.plot(t[0:num-1],M_flap[0:num-1],label='CCBlade')#,label='%sD'%mult[j])
        # ax2.plot(t,M_edge)



    # ax1.plot(t[0:num-1],my_fast[0:num-1],label='FAST')
    # ax2.plot(t,mx_fast)
    atm = my_fast-np.mean(my_fast)
    # atm = (my_fast-np.mean(my_fast))-(M_flap-np.mean(M_flap))
    atm = (my_fast)-(M_flap)
    # atm = M_flap-my_fast
    # ax1.plot(t,my_fast-M_flap/1000.)
    # ax1.plot(t,M_flap+atm,label='both')
    ax1.plot(t[0:num-1],atm[0:num-1],label='atm')

    # ax1.set_ylim(0.,10000.)
    # ax2.set_ylim(0.,1000.)
    ax1.set_title('flapwise')
    # ax2.set_title('edgewise')

    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('root bending moment (kN-m)')
    # ax2.set_xlabel('azimuth angle (radians)')


    # plt.ylim(0.,6000.)

    # plt.figure(2)
    # # plt.plot(t,o)
    # plt.plot(t[0:num-1],s)
    start = TIME.time()
    for j in range(len(mult)):
        print j
        # turbineY = np.array([0.,126.4])*mult[j]
        turbineY = np.array([0.,126.4])*0.5
        # turbineY = np.array([0.,1000000000.])
        for i in range(len(t)-1):
            # az = a_of_t(t[i],Omega)
            current_pos = (current_pos+((o[i+1]+o[i])/2.*(t[i+1]-t[i])/60.)*360.)%360.
            az = current_pos
            # az = (90.*float(i))%360.
            # print 'az: ', az
            # x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY[1],90.,r,yaw_deg,az)
            # speeds = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, wind_speed,shearExp=shearExp)
            # s = np.append(s,speeds[0])
            # # M_flap[i], M_edge[i] = calc_moment(speeds,r[0],r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az,shearExp=shearExp)
            # M_flap[i], M_edge[i] = calc_moment(speeds,r[0],r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,o[i],pitch,azimuth=az,shearExp=shearExp)
            M_flap[i] = f_flap2(az)


    M_flap = M_flap/1000.
    # M_edge = M_edge/1000.
    ax1.plot(t[0:num-1],M_flap[0:num-1],label='CCBlade')
    ax1.plot(t[0:num-1],M_flap[0:num-1]+atm[0:num-1],label='both')



    ax1.legend()

    # plt.savefig('ccblade_moments.pdf',transparent=True)
    plt.legend()


    print max(M_flap[0:num-1])-min(M_flap[0:num-1])

    print diameters
    plt.show()
