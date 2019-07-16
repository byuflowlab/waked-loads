import numpy as np
from ccblade import *
from wakeLoadsFuncs import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as Time
from amount_waked import amount_waked
# import rainflow
import fast_calc_aep
import scipy.signal


from numpy import fabs as fabs
import numpy as np


def rainflow(array_ext,flm=0,l_ult=1e16,uc_mult=0.5):

    # """ Rainflow counting of a signal's turning points with Goodman correction
    #
    #     Args:
    #         array_ext (numpy.ndarray): array of turning points
    #
    #     Keyword Args:
    #         flm (float): fixed-load mean [opt, default=0]
    #         l_ult (float): ultimate load [opt, default=1e16]
    #         uc_mult (float): partial-load scaling [opt, default=0.5]
    #
    #     Returns:
    #         array_out (numpy.ndarray): (5 x n_cycle) array of rainflow values:
    #                                     1) load range
    #                                     2) range mean
    #                                     3) Goodman-adjusted range
    #                                     4) cycle count
    #                                     5) Goodman-adjusted range with flm = 0
    #
    # """

    flmargin = l_ult - fabs(flm)            # fixed load margin
    tot_num = array_ext.size                # total size of input array
    array_out = np.zeros((5, tot_num-1))    # initialize output array

    pr = 0                                  # index of input array
    po = 0                                  # index of output array
    j = -1                                  # index of temporary array "a"
    a  = np.empty(array_ext.shape)          # temporary array for algorithm

    # loop through each turning point stored in input array
    for i in range(tot_num):

        j += 1                  # increment "a" counter
        a[j] = array_ext[pr]    # put turning point into temporary array
        pr += 1                 # increment input array pointer

        while ((j >= 2) & (fabs( a[j-1] - a[j-2] ) <= \
                fabs( a[j] - a[j-1]) ) ):
            lrange = fabs( a[j-1] - a[j-2] )

            # partial range
            if j == 2:
                mean      = ( a[0] + a[1] ) / 2.
                adj_range = lrange * flmargin / ( l_ult - fabs(mean) )
                adj_zero_mean_range = lrange * l_ult / ( l_ult - fabs(mean) )
                a[0]=a[1]
                a[1]=a[2]
                j=1
                if (lrange > 0):
                    array_out[0,po] = lrange
                    array_out[1,po] = mean
                    array_out[2,po] = adj_range
                    array_out[3,po] = uc_mult
                    array_out[4,po] = adj_zero_mean_range
                    po += 1

            # full range
            else:
                mean      = ( a[j-1] + a[j-2] ) / 2.
                adj_range = lrange * flmargin / ( l_ult - fabs(mean) )
                adj_zero_mean_range = lrange * l_ult / ( l_ult - fabs(mean) )
                a[j-2]=a[j]
                j=j-2
                if (lrange > 0):
                    array_out[0,po] = lrange
                    array_out[1,po] = mean
                    array_out[2,po] = adj_range
                    array_out[3,po] = 1.00
                    array_out[4,po] = adj_zero_mean_range
                    po += 1

    # partial range
    for i in range(j):
        lrange    = fabs( a[i] - a[i+1] );
        mean      = ( a[i] + a[i+1] ) / 2.
        adj_range = lrange * flmargin / ( l_ult - fabs(mean) )
        adj_zero_mean_range = lrange * l_ult / ( l_ult - fabs(mean) )
        if (lrange > 0):
            array_out[0,po] = lrange
            array_out[1,po] = mean
            array_out[2,po] = adj_range
            array_out[3,po] = uc_mult
            array_out[4,po] = adj_zero_mean_range
            po += 1

    # get rid of unused entries
    array_out = array_out[:,:po]

    return array_out


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

    return amnt_waked


def setup_airfoil():

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

        tilt = 0.
        precone = 0.
        hubHt = 90.0
        nSector = 1
        pitch = 0.0
        yaw_deg = 0.

        return Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg


def setup_atm(filename_free,filename_close,filename_far,TI=0.11,N=24001):
        print 'read data'
        s = Time.time()

        """free FAST"""
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C676_W8_T11.0_P0.0_m2D_L-1.0/Model.out'
        lines = np.loadtxt(filename_free,skiprows=8)
        time = lines[:,0]
        # mx = lines[:,11]
        my = lines[:,12]
        a = lines[:,5]
        # o_free = lines[:,6]
        Omega_free = np.mean(lines[:,6])
        time = time-time[0]

        m_free = interp1d(time, my, kind='linear')
        a_free = interp1d(time, a, kind='linear')
        # o_free = interp1d(time, o_free, kind='cubic')
        # m_free = my


        """waked FAST CLOSE"""
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        lines = np.loadtxt(filename_close,skiprows=8)
        time = lines[:,0]
        # mx = lines[:,11]
        my = lines[:,12]
        a = lines[:,5]
        # o_waked = lines[:,6]
        Omega_waked = np.mean(lines[:,6])
        time = time-time[0]
        m_close = interp1d(time, my, kind='linear')
        a_close = interp1d(time, a, kind='linear')
        # o_close = interp1d(time, o_waked, kind='cubic')
        # m_close = my

        """waked FAST FAR"""
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
        lines = np.loadtxt(filename_far,skiprows=8)
        time = lines[:,0]
        # mx = lines[:,11]
        my = lines[:,12]
        a = lines[:,5]
        # o_waked = lines[:,6]
        time = time-time[0]
        m_far = interp1d(time, my, kind='linear')
        a_far = interp1d(time, a, kind='linear')
        # o_far = interp1d(time, o_waked, kind='cubic')
        # m_far = my

        print Time.time()-s

        """setup the CCBlade loads"""
        angles = np.linspace(0.,360.,50)

        ccblade_free = np.zeros_like(angles)
        ccblade_close = np.zeros_like(angles)
        ccblade_far = np.zeros_like(angles)

        turbineX_close = np.array([0.,126.4])*4.
        turbineX_far = np.array([0.,126.4])*10.

        turbineY_free = np.array([0.,1264000.])
        turbineY_waked = np.array([0.,0.])

        wind_speed = 8.

        Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()

        print 'make CCBlade functions'
        s = Time.time()
        for i in range(len(angles)):
                az = angles[i]

                #freestream
                x_locs,y_locs,z_locs = findXYZ(turbineX_close[1],turbineY_free[1],90.,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX_close, turbineY_free, x_locs, y_locs, z_locs, wind_speed,TI=TI)
                ccblade_free[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_free,pitch,azimuth=az)

                #waked close
                x_locs,y_locs,z_locs = findXYZ(turbineX_close[1],turbineY_waked[1],90.,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX_close, turbineY_waked, x_locs, y_locs, z_locs, wind_speed,TI=TI)
                ccblade_close[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_free,pitch,azimuth=az)

                #waked far
                x_locs,y_locs,z_locs = findXYZ(turbineX_far[1],turbineY_waked[1],90.,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX_far, turbineY_waked, x_locs, y_locs, z_locs, wind_speed,TI=TI)
                ccblade_far[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_waked,pitch,azimuth=az)

                if i == 0:
                        free_speed = get_eff_turbine_speeds(turbineX_close, turbineY_free, wind_speed,TI=TI)[1]
                        waked_speed = get_eff_turbine_speeds(turbineX_close, turbineY_waked, wind_speed,TI=TI)[1]

        f_free = interp1d(angles, ccblade_free/1000., kind='linear')
        f_close = interp1d(angles, ccblade_close/1000., kind='linear')
        f_far = interp1d(angles, ccblade_far/1000., kind='linear')
        print Time.time()-s

        t = np.linspace(0.,600.,N)

        # t = time
        dt = t[1]-t[0]

        CC_free = np.zeros_like(t)
        CC_close = np.zeros_like(t)
        CC_far = np.zeros_like(t)

        FAST_free = np.zeros_like(t)
        FAST_close = np.zeros_like(t)
        FAST_far = np.zeros_like(t)

        """get atm loads"""

        # pos_free = 0.
        # pos_close = 0.
        # pos_far = 0.

        print 'call CCBlade functions'
        s = Time.time()
        for i in range(len(t)):

                # M_free[i] = f_free(pos_free)
                # pos_free = (pos_free+(o_free((t[i+1]+t[i])/2.)*dt/60.)*360.)%360.
                CC_free[i] = f_free(a_free(t[i]))

                # M_close[i] = f_close(pos_close)
                # pos_close = (pos_close+(o_close((t[i+1]+t[i])/2.)*dt/60.)*360.)%360.
                CC_close[i] = f_close(a_close(t[i]))

                # M_far[i] = f_far(pos_far)
                # pos_far = (pos_far+(o_far((t[i+1]+t[i])/2.)*dt/60.)*360.)%360.
                CC_far[i] = f_far(a_far(t[i]))

                FAST_free[i] = m_free(t[i])
                FAST_close[i] = m_close(t[i])
                FAST_far[i] = m_far(t[i])

        print Time.time()-s

        atm_free = FAST_free-CC_free
        atm_close = FAST_close-CC_close
        atm_far = FAST_far-CC_far
        # atm_free = m_free-CC_free
        # atm_close = m_close-CC_close
        # atm_far = m_far-CC_far

        f_atm_free = interp1d(t, atm_free, kind='linear')
        f_atm_close = interp1d(t, atm_close, kind='linear')
        f_atm_far = interp1d(t, atm_far, kind='linear')

        return f_atm_free,f_atm_close,f_atm_far,Omega_free,Omega_waked,free_speed,waked_speed
        # return atm_free,atm_close,atm_far,time,Omega_free,Omega_waked,free_speed,waked_speed


def get_loads_history(turbineX,turbineY,turb_index,Omega_free,Omega_waked,free_speed,waked_speed,f_atm_free,f_atm_close,f_atm_far,N=24001,TI=0.11,wind_speed=8.,rotor_diameter=126.4):
# def get_loads_history(turbineX,turbineY,turb_index,Omega_free,Omega_waked,free_speed,waked_speed,atm_free,atm_close,atm_far,time,wind_speed=8.,rotor_diameter=126.4,TI=0.11):

    # print 'get loads history'

    # npts = len(time)
    nTurbines = len(turbineX)

    Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()
    angles = np.linspace(0.,360.,100)
    ccblade_moments = np.zeros_like(angles)

    _, wake_radius = get_speeds(turbineX, turbineY, np.array([0.]), np.array([0.]), np.array([0.]), wind_speed, TI=TI)

    # print 'getting CCBlade moments'
    s = Time.time()
    for i in range(len(ccblade_moments)):
        az = angles[i]
        x_locs,y_locs,z_locs = findXYZ(turbineX[turb_index],turbineY[turb_index],90.,r,yaw_deg,az)
        speeds, _ = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, wind_speed,TI=TI)

        if i == 0:
            actual_speed = get_eff_turbine_speeds(turbineX, turbineY, wind_speed,TI=TI)[1]
            Omega = (Omega_waked + (Omega_free-Omega_waked)/(free_speed-waked_speed) * (actual_speed-waked_speed))

        ccblade_moments[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az)

    f_ccblade = interp1d(angles, ccblade_moments/1000., kind='linear')
    # print Time.time()-s

    t = np.linspace(0.,600.,N)
    # t = time
    dt = t[1]-t[0]
    # moments = np.zeros(N)
    moments = np.zeros(N)
    m = np.zeros(N)
    position = 0.

    # s = Time.time()
    # for i in range(N):

    for i in range(N):
        # print '1'
        # s = Time.time()

        amnt_waked = np.zeros(nTurbines)
        dx_dist = np.zeros(nTurbines)
        for waking in range(nTurbines):
            dx = turbineX[turb_index]-turbineX[waking]
            dy = turbineY[turb_index]-turbineY[waking]
            dx_dist[waking] = dx
            if dx < 0.:
                amnt_waked[waking] = 0.
            else:
                amnt_waked[waking] = amount_waked(dy,wake_radius[turb_index][waking]*1.75,rotor_diameter,position)

        waked_array = np.zeros(nTurbines)
        dx_array = np.zeros(nTurbines)

        # print Time.time()-s
        # print '2'
        # s = Time.time()

        num = 0
        indices = np.argsort(dx_dist)
        for waking in range(nTurbines):
            if dx_dist[indices[waking]] > 0.:
                # if num == 0:
                #     if amnt_waked[indices[waking]] > 0.:
                #         waked_array[num] = amnt_waked[indices[waking]]
                #         dx_array[num] = dx_dist[indices[waking]]
                #         num += 1
                # else:
                    if amnt_waked[indices[waking]] > np.sum(waked_array[0:num]):
                        waked_array[num] = amnt_waked[indices[waking]]-np.sum(waked_array[0:num])
                        dx_array[num] = dx_dist[indices[waking]]
                        num += 1

        # print Time.time()-s
        # print '3'
        # s = Time.time()

        down = dx_array/rotor_diameter

        moments[i] = f_ccblade(position)
        m[i] = moments[i]

        unwaked = 1.-np.sum(waked_array)
        # print 'unwaked', unwaked
        for k in range(np.count_nonzero(waked_array)):
            if down[k] < 4.:
                  moments[i] += f_atm_close(t[i])*waked_array[k]
                  # moments[i] += atm_close[i]*waked_array[k]
            elif down[k] > 10.:
                  moments[i] += f_atm_far(t[i])*waked_array[k]
            else:
                  moments[i] += (f_atm_close(t[i])*(10.-down[k])/6.+f_atm_far(t[i])*(down[k]-4.)/6.)*waked_array[k]

        moments[i] += f_atm_free(t[i])*unwaked

        position = (position+(Omega*(dt)/60.)*360.)%360.

        # print Time.time()-s

    # plt.plot(t,f_atm_close(t))
    # plt.plot(t,m)
    if turb_index == 1.:
        plt.plot(t,moments)
    # plt.show()

    return moments


def calc_damage(moments,freq,fos=3):
    N = len(moments)
    t = np.linspace(0.,600.,N)


    d = 0.
    R = 0.5 #root cylinder radius
    I = 0.25*np.pi*R**4
    sigma = moments*R/I

    pp = scipy.signal.find_peaks(sigma)[0]
    pn = scipy.signal.find_peaks(-sigma)[0]
    p = np.append(0,pp)
    p = np.append(p,pn)
    p = np.append(p,len(sigma)-1)
    p = np.sort(p)
    peaks = np.zeros(len(p))
    time = np.zeros(len(p))
    # print 'nPeaks: ', (len(peaks))
    for i in range(len(p)):
        peaks[i] = sigma[p[i]]
        time[i] = t[p[i]]
    array = rainflow(peaks)

    alternate = array[0,:]/2.
    mean = array[1,:]
    count = array[3,:]

    # Goodman correction
    su = 345000.
    mar = alternate/(1.-mean/su)

    npts = len(mar)
    # m = 10.
    for i in range(npts):
        # Nfail = (su/mar[i])**m
        Nfail = 10.**((-mar[i]/su+1.)/0.1)
        # print 'Nfail: ', Nfail
        mult = 25.*365.*24.*6.*freq
        d += count[i]*mult/Nfail


    # plt.plot(t,sigma)
    # plt.plot(time,peaks,'o')
    # plt.xlim(0.,100.)
    # plt.show()

    return d*fos


# def farm_damage(turbineX,turbineY,windDirections,windFrequencies,f_atm_free,f_atm_close,f_atm_far,Omega_free,Omega_waked,free_speed,waked_speed,N=500):
def farm_damage(turbineX,turbineY,windDirections,windFrequencies,atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed,TI=0.11,N=24001):

    damage = np.zeros_like(turbineX)
    nDirections = len(windDirections)
    nTurbines = len(turbineX)
    # for i in range(nDirections):
    #     turbineXw, turbineYw = fast_calc_aep.windframe(windDirections[i], turbineX, turbineY)
    #     for j in range(nTurbines):
    #         moments = get_loads_history(turbineXw,turbineYw,j,Omega_free,Omega_waked,free_speed,waked_speed,f_atm_free,f_atm_close,f_atm_far,N=N)
    #         calc_damage(moments,windFrequencies[i])
    for i in range(nTurbines):
        for j in range(nDirections):
            # print '____________'
            s = Time.time()
            turbineXw, turbineYw = fast_calc_aep.windframe(windDirections[j], turbineX, turbineY)
            # print Time.time()-s
            s = Time.time()
            # moments = get_loads_history(turbineXw,turbineYw,i,Omega_free,Omega_waked,free_speed,waked_speed,f_atm_free,f_atm_close,f_atm_far,N=N)
            moments = get_loads_history(turbineX,turbineY,i,Omega_free,Omega_waked,free_speed,waked_speed,atm_free,atm_close,atm_far,TI=TI,N=N)
            # print Time.time()-s
            s = Time.time()
            damage[i] += calc_damage(moments,windFrequencies[j])
            # print Time.time()-s
    return damage


if __name__ == '__main__':

    # turbineX = np.array([0.,126.4,1.5*126.4])*4.
    # turbineY = np.array([0.,30.,0.])
    # turbineX = np.linspace(0.,4.,5)*126.4*3.
    # turbineY = np.random.rand(5)*300.
    #
    # plt.figure(6)
    # plt.plot(turbineX,turbineY,'o')
    # plt.axis('equal')
    #
    filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
    filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
    filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
    #
    start_setup = Time.time()
    f_atm_free,f_atm_close,f_atm_far,Omega_free,Omega_waked,free_speed,waked_speed = setup_atm(filename_free,filename_close,filename_far)
    print 'setup time: ', Time.time()-start_setup
    #
    # for i in range(len(turbineX)):
    #     turb_index = i
    #     plt.figure(i)
    #     start_run = Time.time()
    #     moments = get_loads_history(turbineX,turbineY,turb_index,Omega_free,Omega_waked,free_speed,waked_speed,f_atm_free,f_atm_close,f_atm_far,N=500)
    #     print 'run time: ', Time.time()-start_run
    #
    #     t = np.linspace(0.,600.,500)
    #     plt.plot(t,moments)
    #     plt.ylim(0.,10000.)
    #     plt.title('%s'%i)
    # plt.show()

    turbineX = np.array([0.,100.,200.,300.])
    turbineY = np.array([0.,100.,0.,-100.])
    windDirections = np.array([0.,90.])
    windFrequencies = np.ones_like(windDirections)/len(windDirections)
    s = Time.time()
    D = farm_damage(turbineX,turbineY,windDirections,windFrequencies,f_atm_free,f_atm_close,f_atm_far,Omega_free,Omega_waked,free_speed,waked_speed,N=500)
    print 'time to run: ', Time.time()-s
    print 'damage: ', D

        #
        # f_atm_free,f_atm_close,f_atm_far = setup_atm(filename_free,filename_close,filename_far)
        # t = np.linspace(0.,600.,1000)
        # plt.plot(t,f_atm_free(t))
        # plt.plot(t,f_atm_close(t))
        # plt.plot(t,f_atm_far(t))
        # plt.show()
    # """waked FAST"""
    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
    #
    # lines = np.loadtxt(filename,skiprows=8)
    # time = lines[:,0]
    # mx = lines[:,11]
    # my = lines[:,12]
    # time = time-time[0]
    #
    # m_t_waked = interp1d(time, my, kind='cubic')
    #
    # t = np.linspace(0.,10.,1000)
    # m = m_t_waked(t)
    #
    # plt.plot(t,m)
    # plt.show()
    #
    # determine_peaks(m)
