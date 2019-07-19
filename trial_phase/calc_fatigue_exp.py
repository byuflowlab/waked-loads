
import numpy as np
import math
from ccblade import *
from wakeLoadsFuncs import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as Time
# from amount_waked import amount_waked
# import rainflow
import fast_calc_aep
import scipy.signal

# import damage_calc


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


def rotor_amount_waked(dy,wake_radius,rotor_diameter):
    """
    find the fraction of the rotor that is waked

    inputs:
    dy:             the crosstream offset between the waked turbine and the waking turbine
    wake_radius:
    rotor_diameter:

    outputs:
    amnt_waked:     the fraction of the rotor that is waked
    """

    r = rotor_diameter/2.
    R = wake_radius

    dy = abs(dy)

    # unwaked
    if dy > (R+r):
        return 0.
    # fully waked
    elif (dy+r) < R and r < R:
        return 1.
    # turbine larger than wake
    elif (dy+R) < r and r > R:
        a_wake = np.pi*R**2
        a_turb = np.pi*r**2
        return a_wake/a_turb

    else:
        p1 = r**2*math.acos((dy**2+r**2-R**2)/(2.*dy*r))
        p2 = R**2*math.acos((dy**2+R**2-r**2)/(2.*dy*R))
        p3 = -0.5*math.sqrt((-dy+r+R)*(dy+r-R)*(dy-r+R)*(dy+r+R))
        a_turb = np.pi*r**2
        return (p1+p2+p3)/a_turb


def amount_waked(dy,wake_radius,rotor_diameter,az):
    """
    find the fraction of the blade that is waked

    inputs:
    dy:             the crosstream offset between the waked turbine and the waking turbine
    wake_radius:
    rotor_diameter:
    az:             the azimuth angle of the blade in question

    outputs:
    amnt_waked:     the fraction of the blade that is waked
    """

    r = rotor_diameter/2.

    if az%360. == 0. or az%360. == 180.:
        m = 1000.
    else:
        m = np.cos(np.deg2rad(az))/np.sin(np.deg2rad(az))
    b = -m*dy

    # the polynomial defining the intersection of the blade line and the wake circle
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
        """
        setup CCBlade inputs
        """

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
        """
        setup the atmospheric turbulence loads that get superimposed on the loads from CCBlade

        inputs:
        filename_free:      the freestream loads FAST file path
        filename_close:     the 4D downstream fully waked loads FAST file path
        filename_far:       the 10D downstream fully waked loads FAST file path
        TI:                 the turbulence intensity
        N:                  this should always be the length of the FAST data you're passing in

        outputs:
        f_atm_free:         a function giving the freestream atmospheric loads as a function of time
        f_atm_close:        a function giving the 4D downstream fully waked atmospheric loads as a function of time
        f_atm_far:          a function giving the 10D downstream fully waked atmospheric loads as a function of time
        Omega_free:         the time average of the rotation rate for the freestream FAST file
        Omega_waked:        the time average of the rotation rate for the close waked FAST file
        free_speed:         the freestream wind speed
        waked_speed:        the fully waked close wind speed
        """

        """free FAST"""
        lines = np.loadtxt(filename_free,skiprows=8)
        time = lines[:,0]
        my = lines[:,12]
        a = lines[:,5]
        Omega_free = np.mean(lines[:,6])
        time = time-time[0]

        m_free = interp1d(time, my, kind='linear')
        a_free = interp1d(time, a, kind='linear')

        """waked FAST CLOSE"""
        lines = np.loadtxt(filename_close,skiprows=8)
        time = lines[:,0]
        my = lines[:,12]
        a = lines[:,5]
        Omega_waked = np.mean(lines[:,6])
        time = time-time[0]

        m_close = interp1d(time, my, kind='linear')
        a_close = interp1d(time, a, kind='linear')

        """waked FAST FAR"""
        lines = np.loadtxt(filename_far,skiprows=8)
        time = lines[:,0]
        my = lines[:,12]
        a = lines[:,5]
        time = time-time[0]
        Omega_waked10 = np.mean(lines[:,6])
        print 'Omega_waked10: ', Omega_waked10

        m_far = interp1d(time, my, kind='linear')
        a_far = interp1d(time, a, kind='linear')


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
                ccblade_close[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_waked,pitch,azimuth=az)

                #waked far
                x_locs,y_locs,z_locs = findXYZ(turbineX_far[1],turbineY_waked[1],90.,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX_far, turbineY_waked, x_locs, y_locs, z_locs, wind_speed,TI=TI)
                ccblade_far[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_waked10,pitch,azimuth=az)

                if i == 0:
                        free_speed = get_eff_turbine_speeds(turbineX_close, turbineY_free, wind_speed,TI=TI)[1]
                        # waked_speed = get_eff_turbine_speeds(turbineX_close, turbineY_waked, wind_speed,TI=TI)[1]
                        waked_speed = get_eff_turbine_speeds(turbineX_far, turbineY_waked, wind_speed,TI=TI)[1]

        f_free = interp1d(angles, ccblade_free/1000., kind='linear')
        f_close = interp1d(angles, ccblade_close/1000., kind='linear')
        f_far = interp1d(angles, ccblade_far/1000., kind='linear')

        t = np.linspace(0.,600.,N)

        dt = t[1]-t[0]

        CC_free = np.zeros_like(t)
        CC_close = np.zeros_like(t)
        CC_far = np.zeros_like(t)

        FAST_free = np.zeros_like(t)
        FAST_close = np.zeros_like(t)
        FAST_far = np.zeros_like(t)

        """get atm loads"""
        print 'call CCBlade functions'
        for i in range(len(t)):
                CC_free[i] = f_free(a_free(t[i]))
                CC_close[i] = f_close(a_close(t[i]))
                CC_far[i] = f_far(a_far(t[i]))

                FAST_free[i] = m_free(t[i])
                FAST_close[i] = m_close(t[i])
                FAST_far[i] = m_far(t[i])

        atm_free = FAST_free-CC_free
        atm_close = FAST_close-CC_close
        atm_far = FAST_far-CC_far

        f_atm_free = interp1d(t, atm_free, kind='linear')
        f_atm_close = interp1d(t, atm_close, kind='linear')
        f_atm_far = interp1d(t, atm_far, kind='linear')

        return f_atm_free,f_atm_close,f_atm_far,Omega_free,Omega_waked10,free_speed,waked_speed


def get_loads_history(turbineX,turbineY,turb_index,Omega_free,Omega_waked,free_speed,waked_speed,f_atm_free,f_atm_close,f_atm_far,N=24001,TI=0.11,wind_speed=8.,rotor_diameter=126.4):
    """
    get the loads history of interpolated FAST data superimposed onto CCBlade loads

    inputs:
    turbineX:       x locations of the turbines (in the wind frame)
    turbineY:       y locations of the turbines (in the wind frame)
    turb_index:     the index of the turbine of interest
    f_atm_free:     a function giving the freestream atmospheric loads as a function of time
    f_atm_close:    a function giving the 4D downstream fully waked atmospheric loads as a function of time
    f_atm_far:      a function giving the 10D downstream fully waked atmospheric loads as a function of time
    Omega_free:     the time average of the rotation rate for the freestream FAST file
    Omega_waked:    the time average of the rotation rate for the close waked FAST file
    free_speed:     the freestream wind speed
    waked_speed:    the fully waked close wind speed

    outputs:
    moments:        the root bending moments time history

    """

    nTurbines = len(turbineX)

    Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()
    angles = np.linspace(0.,360.,100)
    ccblade_moments = np.zeros_like(angles)

    """CCBlade moments"""
    s = Time.time()
    for i in range(len(ccblade_moments)):
        az = angles[i]
        x_locs,y_locs,z_locs = findXYZ(turbineX[turb_index],turbineY[turb_index],90.,r,yaw_deg,az)
        speeds, _ = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, wind_speed,TI=TI)

        if i == 0:
            actual_speed = get_eff_turbine_speeds(turbineX, turbineY, wind_speed,TI=TI)[1]
            Omega = (Omega_waked + (Omega_free-Omega_waked)/(free_speed-waked_speed) * (actual_speed-waked_speed))
            print 'Omega: ', Omega
        ccblade_moments[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az)

    f_ccblade = interp1d(angles, ccblade_moments/1000., kind='linear')

    pos = np.linspace(0.,Omega*10.*360.,N)%360.

    """amount waked"""

    _, sigma = get_speeds(turbineX, turbineY, np.array([0.]), np.array([0.]), np.array([0.]), wind_speed, TI=TI)
    wake_radius = sigma*1.75

    waked_amount = np.zeros((nTurbines,N))
    dx_dist = np.zeros(nTurbines)

    for waking in range(nTurbines):
        dx = turbineX[turb_index]-turbineX[waking]
        dy = turbineY[turb_index]-turbineY[waking]
        dx_dist[waking] = dx
        amnt_waked = np.zeros(len(angles))
        for i in range(len(angles)):
            if dx > 0.:
                amnt_waked[i] = amount_waked(dy,wake_radius[turb_index][waking],rotor_diameter,angles[i])

        waked_func = interp1d(angles, amnt_waked, kind='linear')
        waked_amount[waking,:] = waked_func(pos)

    # rotor_waked = np.zeros(nTurbines)
    # for waking in range(nTurbines):
    #     dy = turbineY[turb_index]-turbineY[waking]
    #     rotor_waked[i] = rotor_amount_waked(dy,wake_radius,rotor_diameter)

    t = np.linspace(0.,600.,N)
    dt = t[1]-t[0]

    moments = f_ccblade(pos)


    #this is the mega time consuming part. It has to loop through this 23000 times :/
    for i in range(N):
        amnt_waked = waked_amount[:,i]

        waked_array = np.zeros(nTurbines)
        dx_array = np.zeros(nTurbines)

        num = 0
        indices = np.argsort(dx_dist)

        #figure out which wakes are the most influential
        for waking in range(nTurbines):
            if dx_dist[indices[waking]] > 0.:
                if amnt_waked[indices[waking]] > np.sum(waked_array[0:num]):
                    waked_array[num] = amnt_waked[indices[waking]]-np.sum(waked_array[0:num])
                    dx_array[num] = dx_dist[indices[waking]]
                    num += 1

        down = dx_array/rotor_diameter
        unwaked = 1.-np.sum(waked_array)

        # down += 1.

        # see how to interpolate the FAST data
        for k in range(np.count_nonzero(waked_array)):
            if down[k] < 4.:
                  moments[i] += f_atm_close(t[i])*waked_array[k]
            elif down[k] > 10.:
                  moments[i] += f_atm_far(t[i])*waked_array[k]
            else:
                  moments[i] += (f_atm_close(t[i])*(10.-down[k])/6.+f_atm_far(t[i])*(down[k]-4.)/6.)*waked_array[k]
        moments[i] += f_atm_free(t[i])*unwaked

    return moments


def calc_damage(moments,freq,fos=3):

    """
    calculate the damage of a turbine from a single direction

    inputs:
    moments:        the moment history
    freq:           the probability of these moments
    fos:            factor of safety


    outputs:
    damage:         fatigue damage

    """

    N = len(moments)
    t = np.linspace(0.,600.,N)


    d = 0.
    R = 0.5 #root cylinder radius
    I = 0.25*np.pi*R**4

    #go from moments to stresses
    sigma = moments*R/I

    #find the peak stresses
    pp = scipy.signal.find_peaks(sigma)[0]
    pn = scipy.signal.find_peaks(-sigma)[0]
    p = np.append(0,pp)
    p = np.append(p,pn)
    p = np.append(p,len(sigma)-1)
    p = np.sort(p)
    peaks = np.zeros(len(p))
    time = np.zeros(len(p))
    for i in range(len(p)):
        peaks[i] = sigma[p[i]]
        time[i] = t[p[i]]

    #rainflow counting
    array = rainflow(peaks)

    alternate = array[0,:]/2.
    mean = array[1,:]
    count = array[3,:]

    # Goodman correction
    su = 345000.
    mar = alternate/(1.-mean/su)

    npts = len(mar)

    #damage calculations
    for i in range(npts):
        # Nfail = (su/mar[i])**m
        Nfail = 10.**((-mar[i]/su+1.)/0.1)
        mult = 25.*365.*24.*6.*freq
        d += count[i]*mult/Nfail

    return d*fos


def farm_damage(turbineX,turbineY,windDirections,windFrequencies,atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed,TI=0.11,N=24001):
    """
    calculate the damage of each turbine in the farm for every wind direction

    inputs:
    turbineX:       x locations of the turbines (in the wind frame)
    turbineY:       y locations of the turbines (in the wind frame)
    windDirections: the wind directions
    windFrequencies: the associated probabilities of the wind directions
    atm_free:       a function giving the freestream atmospheric loads as a function of time
    atm_close:      a function giving the 4D downstream fully waked atmospheric loads as a function of time
    atm_far:        a function giving the 10D downstream fully waked atmospheric loads as a function of time
    Omega_free:     the time average of the rotation rate for the freestream FAST file
    Omega_waked:    the time average of the rotation rate for the close waked FAST file
    free_speed:     the freestream wind speed
    waked_speed:    the fully waked close wind speed


    outputs:
    damage:         fatigue damage of the farm

    """

    damage = np.zeros_like(turbineX)
    nDirections = len(windDirections)
    nTurbines = len(turbineX)

    for i in range(nTurbines):
        for j in range(nDirections):
            turbineXw, turbineYw = fast_calc_aep.windframe(windDirections[j], turbineX, turbineY)
            moments = get_loads_history(turbineX,turbineY,i,Omega_free,Omega_waked,free_speed,waked_speed,atm_free,atm_close,atm_far,TI=TI,N=N)
            damage[i] += calc_damage(moments,windFrequencies[j])
    return damage


if __name__ == '__main__':

    # # T11
    # filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
    # filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
    # filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
    # # T5.6
    # # filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C464_W8_T5.6_P0.0_m2D_L0/Model.out'
    # # filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C437_W8_T5.6_P0.0_4D_L0/Model.out'
    # # filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C455_W8_T5.6_P0.0_10D_L0/Model.out'
    # #
    # N=24001
    # #
    # TI = 0.11
    # print 'setup'
    # s = Time.time()
    # atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed = setup_atm(filename_free,filename_close,filename_far,TI=TI,N=N)
    # print Time.time()-s
    #
    # # sep = np.array([4.,7.,10.])
    # # offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
    #
    # sep = 4.
    # offset = 0.5
    # turbineX = np.array([0.,126.4])*sep
    # turbineY = np.array([0.,126.4])*offset
    # windDirections = np.array([270.])
    # windFrequencies = np.array([1.])
    # s = Time.time()
    # damage = farm_damage(turbineX,turbineY,windDirections,windFrequencies,atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed,TI=TI,N=N)[1]
    # print "time to run: ", (Time.time()-s)/2.
    # print damage
    # # super4 = np.array([0.6465743 , 0.78860154, 0.90094437, 1.00180077, 1.0227498 , 1.00450329, 0.89131835, 0.77691189, 0.6600066 ])

    wake_radius = 40.
    rotor_diameter = 100.
    dy = np.linspace(-200.,200.,10000)
    w = np.zeros_like(dy)
    for i in range(10000):
        w[i] = rotor_amount_waked(dy[i],wake_radius,rotor_diameter)
    plt.plot(dy,w)
    plt.show()
