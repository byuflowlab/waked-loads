
import numpy as np
from ccblade import *
from wakeLoadsFuncs import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as TIME
from amount_waked import amount_waked




if __name__ == '__main__':

    setup_blade = True
    if setup_blade:
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

    setup_turbine = True
    if setup_turbine:
        tilt = 0.
        precone = 0.
        shear_free = 0.0
        shear_wake = 0.0
        shear_actual = 0.0
        hubHt = 90.0
        nSector = 1
        pitch = 0.0
        yaw_deg = 0.

    setup_FAST = True
    if setup_FAST:
            """free FAST"""
            # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C676_W8_T11.0_P0.0_m2D_L-1.0/Model.out'
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'

            lines = np.loadtxt(filename,skiprows=8)
            time = lines[:,0]
            mx = lines[:,11]
            my = lines[:,12]
            o_free = lines[:,6]
            Omega_free = np.mean(lines[:,6])
            free_speed = np.mean(lines[:,1])
            time = time-time[0]

            # plt.plot(time,my)
            # plt.ylim(0.,10000.)
            # plt.show()

            m_free = interp1d(time, my, kind='cubic')
            o_free = interp1d(time, o_free, kind='cubic')

            """waked FAST CLOSE"""
            # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'

            lines = np.loadtxt(filename,skiprows=8)
            time = lines[:,0]
            mx = lines[:,11]
            my = lines[:,12]
            o_waked = lines[:,6]
            Omega_waked = np.mean(lines[:,6])
            time = time-time[0]
            m_close = interp1d(time, my, kind='cubic')
            o_close = interp1d(time, o_waked, kind='cubic')

            """waked FAST FAR"""
            filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'

            lines = np.loadtxt(filename,skiprows=8)
            time = lines[:,0]
            mx = lines[:,11]
            my = lines[:,12]
            o_waked = lines[:,6]
            time = time-time[0]
            m_far = interp1d(time, my, kind='cubic')
            o_far = interp1d(time, o_waked, kind='cubic')

    setup_atm = True
    if setup_atm:
            """setup the CCBlade loads"""
            angles = np.linspace(0.,360.,100)

            ccblade_free = np.zeros_like(angles)
            ccblade_close = np.zeros_like(angles)
            ccblade_far = np.zeros_like(angles)

            turbineX_close = np.array([0.,126.4])*4.
            turbineX_far = np.array([0.,126.4])*10.

            turbineY_free = np.array([0.,1264000.])
            turbineY_waked = np.array([0.,0.])

            wind_speed = 8.

            for i in range(len(angles)):
                az = angles[i]

                #freestream
                x_locs,y_locs,z_locs = findXYZ(turbineX_close[1],turbineY_free[1],90.,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX_close, turbineY_free, x_locs, y_locs, z_locs, wind_speed,shearExp=shear_free)
                ccblade_free[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_free,pitch,azimuth=az,shearExp=shear_free)

                #waked close
                x_locs,y_locs,z_locs = findXYZ(turbineX_close[1],turbineY_waked[1],90.,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX_close, turbineY_waked, x_locs, y_locs, z_locs, wind_speed,shearExp=shear_free)
                ccblade_close[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_free,pitch,azimuth=az,shearExp=shear_free)

                #waked far
                x_locs,y_locs,z_locs = findXYZ(turbineX_far[1],turbineY_waked[1],90.,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX_far, turbineY_waked, x_locs, y_locs, z_locs, wind_speed,shearExp=shear_wake)
                ccblade_far[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_waked,pitch,azimuth=az,shearExp=shear_wake)

                if i == 0:
                    free_speed = get_eff_turbine_speeds(turbineX_close, turbineY_free, wind_speed)[1]
                    waked_speed = get_eff_turbine_speeds(turbineX_close, turbineY_waked, wind_speed)[1]

            f_free = interp1d(angles, ccblade_free/1000., kind='cubic')
            f_close = interp1d(angles, ccblade_close/1000., kind='cubic')
            f_far = interp1d(angles, ccblade_far/1000., kind='cubic')

            t = np.linspace(0.,600.,1000)
            dt = t[1]-t[0]
            M_free = np.zeros_like(t)
            M_close = np.zeros_like(t)
            M_far = np.zeros_like(t)


            FAST_free = np.zeros_like(t)
            FAST_close = np.zeros_like(t)
            FAST_far = np.zeros_like(t)

            """get atm loads"""

            pos_free = 0.
            pos_close = 0.
            pos_far = 0.

            for i in range(len(t)-1):

                M_free[i] = f_free(pos_free)
                pos_free = (pos_free+(o_free((t[i+1]+t[i])/2.)*dt/60.)*360.)%360.

                M_close[i] = f_close(pos_close)
                pos_close = (pos_close+(o_close((t[i+1]+t[i])/2.)*dt/60.)*360.)%360.

                M_far[i] = f_far(pos_far)
                pos_far = (pos_far+(o_far((t[i+1]+t[i])/2.)*dt/60.)*360.)%360.

                FAST_free[i] = m_free(t[i])
                FAST_close[i] = m_close(t[i])
                FAST_far[i] = m_far(t[i])


            atm_free = FAST_free-M_free
            atm_close = FAST_close-M_close
            atm_far = FAST_far-M_far

            f_atm_free = interp1d(t, atm_free, kind='cubic')
            f_atm_close = interp1d(t, atm_close, kind='cubic')
            f_atm_far = interp1d(t, atm_far, kind='cubic')

            # plt.plot(t[0:-1],FAST_free[0:-1],'k',label='FAST')
            # plt.plot(t[0:-1],M_free[0:-1],'r',label='CCBlade')
            # plt.plot(t[0:-1],atm_free[0:-1],'b', label='atm')

            # plt.plot(t[0:-1],FAST_close[0:-1],'k',label='FAST')
            # plt.plot(t[0:-1],M_close[0:-1],'r',label='CCBlade')
            # plt.plot(t[0:-1],atm_close[0:-1],'b', label='atm')
            #
            # plt.plot(t[0:-1],FAST_far[0:-1],'k',label='FAST')
            # plt.plot(t[0:-1],M_far[0:-1],'r',label='CCBlade')
            # plt.plot(t[0:-1],atm_far[0:-1],'b', label='atm')

            # plt.title('freestream')
            # plt.xlabel('time (s)')
            # plt.ylabel('root bending moment (kN-m)')
            # plt.gca().set_ylim(-2000.,10000.)
            # #
            # plt.legend()
            # plt.show()


    delt = np.array([-1.0,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.0])
    down = 7.
    turbineX = np.array([0.,126.4])*down
    rotor_diameter = 126.4

    for k in range(len(delt)):
        ccblade_actual = np.zeros_like(angles)
        D = delt[k]
        dy = rotor_diameter*D

        turbineY_actual = np.array([0.,dy])

        for i in range(len(angles)):
            az = angles[i]
            x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY_actual[1],90.,r,yaw_deg,az)
            speeds, wake_radius = get_speeds(turbineX, turbineY_actual, x_locs, y_locs, z_locs, wind_speed,shearExp=shear_actual)

            if i == 0:
                actual_speed = get_eff_turbine_speeds(turbineX, turbineY_actual, wind_speed)[1]
                Omega_actual = (Omega_waked + (Omega_free-Omega_waked)/(free_speed-waked_speed) * (actual_speed-waked_speed))

            ccblade_actual[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_actual,pitch,azimuth=az,shearExp=shear_actual)

        f_actual = interp1d(angles, ccblade_actual/1000., kind='cubic')

        M_actual = np.zeros(len(t))
        M_actual_turb = np.zeros_like(M_actual)

        pos_actual = 0.

        plt.figure(k)
        for i in range(len(M_actual)):
            waked = amount_waked(dy,wake_radius[1][0],rotor_diameter,pos_actual)
            M_actual[i] = f_actual(pos_actual)

            if down < 4.:
                    M_actual_turb[i] = f_actual(pos_actual) + f_atm_close(t[i])*waked + f_atm_free(t[i])*(1.-waked)
            elif down > 10.:
                    M_actual_turb[i] = f_actual(pos_actual) + f_atm_far(t[i])*waked + f_atm_free(t[i])*(1.-waked)
            else:
                    M_actual_turb[i] = f_actual(pos_actual) + (f_atm_close(t[i])*(10.-down)/6.+f_atm_far(t[i])*(down-4.)/6.)*waked + f_atm_free(t[i])*(1.-waked)
            pos_actual = (pos_actual+(Omega_actual*(dt)/60.)*360.)%360.


        plt.plot(t,M_actual_turb,label='superimposed')
        # plt.plot(t,M_actual,label='CCBlade')


        if down == 4.:
                """partial waked FAST"""
                if D == 0.:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
                elif D == -0.25:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C652_W8_T11.0_P0.0_4D_L-0.25/Model.out'
                elif D == 0.25:
                    filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C654_W8_T11.0_P0.0_4D_L0.25/Model.out'
                elif D == -0.5:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C651_W8_T11.0_P0.0_4D_L-0.5/Model.out'
                elif D == 0.5:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C655_W8_T11.0_P0.0_4D_L0.5/Model.out'
                elif D == -0.75:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C650_W8_T11.0_P0.0_4D_L-0.75/Model.out'
                elif D == 0.75:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C656_W8_T11.0_P0.0_4D_L0.75/Model.out'
                elif D == -1.0:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C649_W8_T11.0_P0.0_4D_L-1.0/Model.out'
                elif D == 1.0:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C657_W8_T11.0_P0.0_4D_L1.0/Model.out'

        if down == 7.:
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


        if down == 10.:
                """partial waked FAST"""
                if D == 0.:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
                elif D == -0.25:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C670_W8_T11.0_P0.0_10D_L-0.25/Model.out'
                elif D == 0.25:
                    filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C672_W8_T11.0_P0.0_10D_L0.25/Model.out'
                elif D == -0.5:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C669_W8_T11.0_P0.0_10D_L-0.5/Model.out'
                elif D == 0.5:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C673_W8_T11.0_P0.0_10D_L0.5/Model.out'
                elif D == -0.75:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C668_W8_T11.0_P0.0_10D_L-0.75/Model.out'
                elif D == 0.75:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C674_W8_T11.0_P0.0_10D_L0.75/Model.out'
                elif D == -1.0:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C667_W8_T11.0_P0.0_10D_L-1.0/Model.out'
                elif D == 1.0:
                    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C675_W8_T11.0_P0.0_10D_L1.0/Model.out'


        if down == 4. or down==7. or down==10.:
                lines = np.loadtxt(filename,skiprows=8)
                time = lines[:,0]
                mx = lines[:,11]
                my = lines[:,12]
                time = time-time[0]
                m_t_p = interp1d(time, my, kind='cubic')
                plt.plot(t,m_t_p(t),label='FAST')

        plt.legend()
        plt.title('%s'%D)
        plt.ylim(0.,10000.)

plt.show()
