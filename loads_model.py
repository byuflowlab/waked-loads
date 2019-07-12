
import numpy as np
from ccblade import *
from wakeLoadsFuncs import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as TIME
from amount_waked import amount_waked




if __name__ == '__main__':

    start_setup = TIME.time()
    """blade geometry"""
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
    shear_free = 0.0
    shear_wake = 0.0
    shear_actual = 0.0
    hubHt = 90.0
    nSector = 1

    tsr = 7.55
    pitch = 0.0
    Omega_free = 11.5
    Omega_waked = 7.5

    # turbineX = np.array([0.,505.6])
    turbineX = np.array([0.,126.4])*8.
    wind_speed = 8.

    yaw_deg = 0.

    """free FAST"""
    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/W8_11.0.out'
    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C657_W8_T11.0_P0.0_4D_L1.0/Model.out'
    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C649_W8_T11.0_P0.0_4D_L-1.0/Model.out'

    lines = np.loadtxt(filename,skiprows=8)
    time = lines[:,0]
    mx = lines[:,19]
    my = lines[:,20]
    # mx = lines[:,11]
    # my = lines[:,12]
    # o_free = lines[:,9]
    o_free = lines[:,6]
    # Omega_free = np.mean(lines[:,9])
    Omega_free = np.mean(lines[:,6])
    free_speed = np.mean(lines[:,1])

    print Omega_free

    u = lines[:,1]
    v = lines[:,2]

    time = time-time[0]

    tsr = o_free*(126.4/2.)/np.sqrt(u**2+v**2)
    # plt.plot(time,tsr,label='free')
    m_t_free = interp1d(time, my, kind='cubic')
    o_t_free = interp1d(time, o_free, kind='cubic')

    """waked FAST"""
    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'

    lines = np.loadtxt(filename,skiprows=8)
    time = lines[:,0]
    mx = lines[:,11]
    my = lines[:,12]
    o_waked = lines[:,6]
    Omega_waked = np.mean(lines[:,6])
    waked_speed = np.mean(lines[:,1])
    print Omega_waked

    az_deg = lines[:,5]

    u = lines[:,1]
    v = lines[:,2]

    time = time-time[0]



    # tsr = o_waked*(126.4/2.)/np.sqrt(u**2+v**2)
    # plt.plot(time,tsr,label='waked')
    #
    # plt.legend()
    # plt.show()

    m_t_waked = interp1d(time, my, kind='cubic')
    o_t_waked = interp1d(time, o_waked, kind='cubic')
    a_t_waked = interp1d(time, az_deg, kind='cubic')


    """setup the CCBlade loads"""
    angles = np.linspace(0.,360.,100)

    ccblade_flap_free = np.zeros_like(angles)
    ccblade_flap_waked = np.zeros_like(angles)


    turbineY_free = np.array([0.,126.4])
    turbineY_waked = np.array([0.,0.])


    turbineX = np.array([0.,126.4])*4.
    for i in range(len(angles)):
        az = angles[i]

        x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY_free[1],90.,r,yaw_deg,az)
        speeds, _ = get_speeds(turbineX, turbineY_free, x_locs, y_locs, z_locs, wind_speed,shearExp=shear_free,wec_factor=1.25)
        ccblade_flap_free[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_free,pitch,azimuth=az,shearExp=shear_free)

        x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY_waked[1],90.,r,yaw_deg,az)
        speeds, _ = get_speeds(turbineX, turbineY_waked, x_locs, y_locs, z_locs, wind_speed,shearExp=shear_wake,wec_factor=1.25)
        ccblade_flap_waked[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_waked,pitch,azimuth=az,shearExp=shear_wake)

        if i == 0:
            free_speed = get_eff_turbine_speeds(turbineX, turbineY_free, wind_speed)[1]
            waked_speed = get_eff_turbine_speeds(turbineX, turbineY_waked, wind_speed)[1]


    print 'free_speed: ', free_speed
    print 'waked_speed: ', waked_speed
    f_free = interp1d(angles, ccblade_flap_free/1000., kind='cubic')
    f_waked = interp1d(angles, ccblade_flap_waked/1000., kind='cubic')


    t = np.linspace(0.,100.,2000)
    dt = t[1]-t[0]
    M_free = np.zeros_like(t)
    M_waked = np.zeros_like(t)


    FAST_free = np.zeros_like(t)
    FAST_waked = np.zeros_like(t)

    """get atm loads"""

    pos_free = 0.
    pos_waked = 0.

    for i in range(len(t)-1):

        # pos_free = (pos_free+(Omega_free*p1*(t[1]-t[0])/60.)*360.)%360.
        M_free[i] = f_free(pos_free)
        pos_free = (pos_free+(o_t_free((t[i+1]+t[i])/2.)*dt/60.)*360.)%360.


        # pos_waked = (pos_waked+(Omega_waked*p2*(t[1]-t[0])/60.)*360.)%360.
        M_waked[i] = f_waked(pos_waked)
        pos_waked = (pos_waked+(o_t_waked((t[i+1]+t[i])/2.)*dt/60.)*360.)%360.



        FAST_free[i] = m_t_free(t[i])
        FAST_waked[i] = m_t_waked(t[i])

    star = 10
    # atm_free = FAST_free[star:-1]-M_free[star:-1]
    # atm_waked = FAST_waked[star:-1]-M_waked[star:-1]
    atm_free = FAST_free-M_free
    atm_waked = FAST_waked-M_waked

    f_atm_free = interp1d(t, atm_free, kind='cubic')
    # f_atm_waked = interp1d(t[star:-1], atm_waked, kind='cubic')
    f_atm_waked = interp1d(t, atm_waked, kind='cubic')

    # t = t[star:-1]



    # plt.plot(t,M_free,'--k', label='CCBlade')
    plt.plot(t,FAST_free, '-k', label='freestream')
    # plt.plot(t,M_free,'--r')
    # plt.plot(t,FAST_free,'-r')
    # plt.plot(t,atm_free,'-r', label='freestream')

    # plt.plot(t,M_waked,'--b')
    plt.plot(t,FAST_waked,'-b', label='waked')
    # plt.plot(t,atm_waked,'-b', label='waked')

    plt.xlabel('time (s)')
    plt.ylabel('root bending moment (kN-m)')
    plt.gca().set_ylim(0.,10000.)

    plt.legend()
    plt.show()

    # print 'setup time: ', TIME.time()-start_setup

    om_mult = np.linspace(4.,6.,21)
    delt = np.array([-1.0,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.0])
    # for j in range(len(om_mult)):
    #     rms = np.zeros(len(delt))

    turbineX = np.array([0.,126.4])*7.
    for k in range(len(delt)):
        # print k
        start_run = TIME.time()
        ccblade_flap_actual = np.zeros_like(angles)
        turbineX_actual = np.array([0.,505.6])
        rotor_diameter = 126.4
        D = delt[k]
        dy = rotor_diameter*D
        turbineY_actual = np.array([0.,dy])

        for i in range(len(angles)):
            az = angles[i]

            x_locs,y_locs,z_locs = findXYZ(turbineX_actual[1],turbineY_actual[1],90.,r,yaw_deg,az)
            speeds, wake_radius = get_speeds(turbineX_actual, turbineY_actual, x_locs, y_locs, z_locs, wind_speed,shearExp=shear_actual,wec_factor=1.)

            if i == 0:
                actual_speed = get_eff_turbine_speeds(turbineX_actual, turbineY_actual, wind_speed,wec_factor=1.25)[1]
                # Omega_actual = (Omega_waked + (Omega_free-Omega_waked)/(free_speed-waked_speed) * (actual_speed-waked_speed))*4.9/6.*0.8
                Omega_actual = (Omega_waked + (Omega_free-Omega_waked)/(free_speed-waked_speed) * (actual_speed-waked_speed))
                print 'delt: ', delt[k]
                print 'actual_speed: ', actual_speed
                print 'Omega_actual: ', Omega_actual
                # Omega_actual = (Omega_waked + (Omega_free-Omega_waked)/(free_speed-waked_speed) * (actual_speed-waked_speed))*11./12.

            ccblade_flap_actual[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_actual,pitch,azimuth=az,shearExp=shear_actual)

        f_actual = interp1d(angles, ccblade_flap_actual/1000., kind='cubic')
        # M_actual = np.zeros(len(t)-star-1)
        M_actual = np.zeros(len(t))
        M_actual_turb = np.zeros_like(M_actual)

        pos_actual = 0.

        plt.figure(k)
        # print 'dy: ', dy
        # print 'wake_radius: ', wake_radius
        # print 'rotor_diameter: ', rotor_diameter
        for i in range(len(M_actual)):
        # for i in range(5):
            # pos = (a_t_waked(t[i])+50.)%360.
            # print pos
            # M_actual[i] = f_actual(pos)
            # M_actual_turb[i] = f_actual(pos) + f_atm_waked(t[i])

            waked = amount_waked(dy,wake_radius[1][0],rotor_diameter,pos_actual)#/amount_waked(126.4,wake_radius[1][0],rotor_diameter,pos_actual)
            # waked = 1.-abs(dy)
            if waked < 0.:
                waked = 0.
            M_actual[i] = f_actual(pos_actual)
            # M_actual_turb[i] = f_actual(pos_actual) + waked*f_atm_waked(t[i+star]) + (1.-waked)*f_atm_free(t[i+star])
            # M_actual_turb[i] = f_actual(pos_actual) + f_atm_waked(t[i+star])
            # M_actual_turb[i] = f_actual(pos_actual) + f_atm_waked(t[i])#*(waked*0.5+0.5)
            M_actual_turb[i] = f_actual(pos_actual) + f_atm_waked(t[i])*waked + f_atm_free(t[i])*(1.-waked)
            pos_actual = (pos_actual+(Omega_actual*(dt)/60.)*360.)%360.

        # print 'run time: ', TIME.time()-start_run
        # t = t[star:-1]
        # M_actual = M_actual[star:-1]

        plt.plot(t,M_actual_turb,label='superimposed')
        # plt.plot(t,M_actual,label='CCBlade')


        # """partial waked FAST"""
        # if D == 0.:
        #     filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        # elif D == -0.25:
        #     filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C652_W8_T11.0_P0.0_4D_L-0.25/Model.out'
        # elif D == 0.25:
        #     filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C654_W8_T11.0_P0.0_4D_L0.25/Model.out'
        # elif D == -0.5:
        #     filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C651_W8_T11.0_P0.0_4D_L-0.5/Model.out'
        # elif D == 0.5:
        #     filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C655_W8_T11.0_P0.0_4D_L0.5/Model.out'
        # elif D == -0.75:
        #     filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C650_W8_T11.0_P0.0_4D_L-0.75/Model.out'
        # elif D == 0.75:
        #     filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C656_W8_T11.0_P0.0_4D_L0.75/Model.out'
        # elif D == -1.0:
        #     filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C649_W8_T11.0_P0.0_4D_L-1.0/Model.out'
        # elif D == 1.0:
        #     filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C657_W8_T11.0_P0.0_4D_L1.0/Model.out'

        # 7 D downstream
        """partial waked FAST"""
        if D == 0.:
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C662_W8_T11.0_P0.0_7D_L0/Model.out'
        elif D == -0.25:
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C661_W8_T11.0_P0.0_7D_L-0.25/Model.out'
        elif D == 0.25:
            filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C663_W8_T11.0_P0.0_7D_L0.25/Model.out'
        elif D == -0.5:
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C660_W8_T11.0_P0.0_7D_L-0.5/Model.out'
        elif D == 0.5:
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
        elif D == -0.75:
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C659_W8_T11.0_P0.0_7D_L-0.75/Model.out'
        elif D == 0.75:
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C665_W8_T11.0_P0.0_7D_L0.75/Model.out'
        elif D == -1.0:
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C658_W8_T11.0_P0.0_7D_L-1.0/Model.out'
        elif D == 1.0:
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C666_W8_T11.0_P0.0_7D_L1.0/Model.out'


        lines = np.loadtxt(filename,skiprows=8)
        time = lines[:,0]
        mx = lines[:,11]
        my = lines[:,12]
        o_waked = lines[:,6]

        time = time-time[0]
        m_t_p = interp1d(time, my, kind='cubic')

        # rms[k] = np.sum((M_actual_turb-m_t_p(t))**2)

        plt.plot(t,m_t_p(t),label='FAST')

        # plt.plot(t,atm_free)
        # plt.plot(t,atm_waked)
        # plt.plot(t,M_actual)
        plt.legend()
        plt.title('%s'%D)
        plt.ylim(0.,10000.)

    # print 'omega multiplier: ', om_mult[j]
    # print 'rms: ', rms
    # print 'sum rms: ', sum(rms)
    # plt.plot(om_mult[j],sum(rms),'o')
    # plt.pause(0.01)

plt.show()
