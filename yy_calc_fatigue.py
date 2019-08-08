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


def find_omega(filename_free,filename_close,filename_far,TI=0.11,wind_speed=8.):
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
        Omega_free = np.mean(lines[:,6])

        """waked FAST CLOSE"""
        lines = np.loadtxt(filename_close,skiprows=8)
        Omega_close = np.mean(lines[:,6])

        """waked FAST FAR"""
        lines = np.loadtxt(filename_far,skiprows=8)
        Omega_far = np.mean(lines[:,6])


        """setup the CCBlade loads"""
        turbineX_close = np.array([0.,126.4])*4.
        turbineX_far = np.array([0.,126.4])*10.

        turbineY_waked = np.array([0.,0.])

        hub_height = 90.

        free_speed = wind_speed
        close_speed = get_eff_turbine_speeds(turbineX_close, turbineY_waked, wind_speed,TI=TI)[1]
        far_speed = get_eff_turbine_speeds(turbineX_far, turbineY_waked, wind_speed,TI=TI)[1]

        print 'tip speed ratios'
        print 'free: ', Omega_free*126.4/2./free_speed
        print 'close: ', Omega_close*126.4/2./close_speed
        print 'far: ', Omega_far*126.4/2./far_speed

        # return flap_free_atm, edge_free_atm, Omega_free, free_speed, flap_close_atm, edge_close_atm, Omega_close, close_speed, flap_far_atm, edge_far_atm, Omega_far, far_speed
        return Omega_free, free_speed, Omega_close, close_speed, Omega_far, far_speed


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

        return Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg


def make_edge_loads(upper,lower,Omega,N=24001,duration=10.):
        n_cycles = Omega*duration
        t = np.linspace(0.,float(n_cycles),N)
        amp = (upper-lower)/2.
        avg = (upper+lower)/2.
        m = np.sin(t*2.*np.pi)*amp+avg
        return m


def get_edgewise_damage(turbineX,turbineY,turb_index,Omega_free,free_speed,Omega_close,close_speed,Omega_far,far_speed,
                        Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=0.11):

        o = np.array([Omega_free,Omega_far,Omega_close,Omega_close])
        sp = np.array([free_speed,far_speed,close_speed,0.])
        f_o = interp1d(sp,o,kind='linear')
        actual_speed = get_eff_turbine_speeds(turbineX, turbineY, free_speed,TI=TI)[turb_index]
        Omega = f_o(actual_speed)
        # print 'Omega: ', Omega


        az = 90.
        x_locs,y_locs,z_locs = findXYZ(turbineX[turb_index],turbineY[turb_index],hubHt,r,yaw_deg,az)
        # print 'Omega: ', Omega
        # print 'turbineX: ', turbineX
        # print 'turbineY: ', turbineY
        # print 'x_locs: ', x_locs
        # print 'y_locs: ', y_locs
        # print 'z_locs: ', z_locs
        # print 'free_speed: ', free_speed
        # print 'TI: ', TI
        speeds, _ = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, free_speed, TI=TI)
        # edge90 = calc_moment_edge(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,wind_speed,pitch,azimuth=az)
        _,edge90 = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az)
        print 'edge90: ', edge90
        az = 270.
        x_locs,y_locs,z_locs = findXYZ(turbineX[turb_index],turbineY[turb_index],hubHt,r,yaw_deg,az)
        speeds, _ = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, free_speed, TI=TI)
        # edge270 = calc_moment_edge(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,wind_speed,pitch,azimuth=az)
        _,edge270 = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az)

        cycle = np.array([edge90,edge270])/1000.

        moment_mean = np.sum(cycle)/2.
        moment_alternate = (np.max(cycle)-np.min(cycle))/2.

        R = 0.5 #root cylinder radius
        I = 0.25*np.pi*(R**4-(R-0.08)**4)

        mean = (moment_mean*R/I)
        alternate = (moment_alternate*R/I)

        # Goodman correction
        # su = 345000.
        su = 4590000.
        effective = alternate/(1.-mean/su)

        m = 10.
        # Nfail = 10.**((-mar[i]/su+1.)/0.1) #I'm pretty sure this is wrong for these cycles
        Nfail = (su/effective)**m #mLife

        nCycles = Omega*60.*24.*365.25*20.

        d = nCycles/Nfail

        return d


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


def calc_damage_moments(m_edge,freq,fos=2):

    """
    calculate the damage of a turbine from a single direction

    inputs:
    moments:        the moment history
    freq:           the probability of these moments
    fos:            factor of safety

    outputs:
    damage:         fatigue damage

    """

    d = 0.
    R = 0.5 #root cylinder radius
    I = 0.25*np.pi*(R**4-(R-0.08)**4)

    sigma = m_edge*R/I

    #find the peak stresses
    pp = scipy.signal.find_peaks(sigma)[0]
    pn = scipy.signal.find_peaks(-sigma)[0]
    p = np.append(0,pp)
    p = np.append(p,pn)
    p = np.append(p,len(sigma)-1)
    p = np.sort(p)
    peaks = np.zeros(len(p))
    vv = np.arange(len(sigma))
    v = np.zeros(len(p))

    for i in range(len(p)):
        peaks[i] = sigma[p[i]]
        v[i] = vv[p[i]]

    #rainflow counting
    array = rainflow(peaks)

    alternate = array[0,:]/2.
    mean = array[1,:]
    count = array[3,:]

    # Goodman correction
    # su = 345000.
    su = 459000.
    mar = alternate/(1.-mean/su)

    npts = len(mar)

    # plt.plot(count,mar,'o')
    # plt.show()

    #damage calculations
    n = np.zeros(npts)
    m = 10.
    for i in range(npts):
        # Nfail = 10.**((-mar[i]/su+1.)/0.1)
        Nfail = ((su)/(mar[i]))**m
        n[i] = Nfail
        mult = 20.*365.*24.*6.*freq

        d += count[i]*mult/Nfail
        # if count[i]*mult/Nfail > 0.02:
        #         print Nfail


    # plt.plot(count,n,'o')
    # plt.show()


    return d


def farm_damage(turbineX,turbineY,windDirections,windFrequencies,Omega_free,free_speed,Omega_close,close_speed,Omega_far,far_speed,
                        Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=0.11):
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

    for j in range(nDirections):
            turbineXw, turbineYw = fast_calc_aep.windframe(windDirections[j], turbineX, turbineY)
            for i in range(nTurbines):
                damage[i] += get_edgewise_damage(turbineXw,turbineYw,i,Omega_free,free_speed,Omega_close,close_speed,Omega_far,far_speed,
                                        Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI)*windFrequencies[j]
    return damage


if __name__ == '__main__':

        filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C655_W8_T11.0_P0.0_4D_L0.5/Model.out'

        lines = np.loadtxt(filename,skiprows=8)
        angles = lines[:,5]
        mx_FAST = lines[:,11]

        # angles = angles[0:1000]
        # mx_FAST = mx_FAST[0:1000]

        ang = np.array([])
        mom = np.array([])
        for i in range(1000):
                ang = np.append(ang,angles[i+37])
                mom = np.append(mom,mx_FAST[i+37])
                if angles[i+38] < angles[i+37]:
                        plt.plot(ang,mom,color='C0')
                        ang = np.array([])
                        mom = np.array([])

        Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()

        hub_height = 90.

        # angles = np.linspace(0.,720.,100)
        edge1 = np.zeros_like(angles)
        flap1 = np.zeros_like(angles)

        edge2 = np.zeros_like(angles)
        flap2 = np.zeros_like(angles)

        turbineX = np.array([0.,126.4])*7.

        turbineY1 = np.array([0.,126.4])*0.5

        angles = np.linspace(0.,360.,100)
        edge1 = np.zeros_like(angles)
        flap1 = np.zeros_like(angles)
        for i in range(len(angles)):

                az = angles[i]
                #freestream
                x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY1[1],hub_height,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX, turbineY1, x_locs, y_locs, z_locs, 8.0,TI=0.11)
                flap1[i], edge1[i] = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)

        plt.plot(angles,edge1/1000.,color='C1',linewidth=2)



        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
        filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C655_W8_T11.0_P0.0_4D_L0.5/Model.out'

        lines = np.loadtxt(filename,skiprows=8)
        angles = lines[:,5]
        mx_FAST = lines[:,11]

        # angles = angles[0:1000]
        # mx_FAST = mx_FAST[0:1000]

        ang = np.array([])
        mom = np.array([])
        for i in range(1000):
                ang = np.append(ang,angles[i+37])
                mom = np.append(mom,mx_FAST[i+37])
                if angles[i+38] < angles[i+37]:
                        plt.plot(ang,mom,color='C0')
                        ang = np.array([])
                        mom = np.array([])

        Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()

        hub_height = 90.

        # angles = np.linspace(0.,720.,100)
        edge1 = np.zeros_like(angles)
        flap1 = np.zeros_like(angles)

        edge2 = np.zeros_like(angles)
        flap2 = np.zeros_like(angles)

        turbineX = np.array([0.,126.4])*4.

        turbineY1 = np.array([0.,126.4])*0.5

        angles = np.linspace(0.,360.,100)
        edge1 = np.zeros_like(angles)
        flap1 = np.zeros_like(angles)
        for i in range(len(angles)):

                az = angles[i]
                #freestream
                x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY1[1],hub_height,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX, turbineY1, x_locs, y_locs, z_locs, 8.0,TI=0.11)
                flap1[i], edge1[i] = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)

        plt.plot(angles,edge1/1000.,color='C1',linewidth=2)




        plt.show()

                # x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY2[1],hub_height,r,yaw_deg,az)
                # speeds, _ = get_speeds(turbineX, turbineY2, x_locs, y_locs, z_locs, 8.0,TI=0.11)
                # flap2[i], edge2[i] = calc_moment(speeds,Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)

        # plt.plot(angles,flap1/1000.,'--r')

        # az = 0.
        # x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY1[1],hub_height,r,yaw_deg,az)
        # speeds, _ = get_speeds(turbineX, turbineY1, x_locs, y_locs, z_locs, 8.0,TI=0.11)
        # print calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)[1]/1000.
        # plt.plot(az,calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)[1]/1000.,'ok')
        #
        # az = 90.
        # x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY1[1],hub_height,r,yaw_deg,az)
        # speeds, _ = get_speeds(turbineX, turbineY1, x_locs, y_locs, z_locs, 8.0,TI=0.11)
        # print calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)[1]/1000.
        # plt.plot(az,calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)[1]/1000.,'ok')
        #
        # az = 180.
        # x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY1[1],hub_height,r,yaw_deg,az)
        # speeds, _ = get_speeds(turbineX, turbineY1, x_locs, y_locs, z_locs, 8.0,TI=0.11)
        # print calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)[1]/1000.
        # plt.plot(az,calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)[1]/1000.,'ok')
        #
        # az = 270.
        # x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY1[1],hub_height,r,yaw_deg,az)
        # speeds, _ = get_speeds(turbineX, turbineY1, x_locs, y_locs, z_locs, 8.0,TI=0.11)
        # print calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)[1]/1000.
        # plt.plot(az,calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)[1]/1000.,'ok')


        # plt.plot(angles,edge1/1000.,color='C1')

        # plt.plot(angles,mx_FAST)

        # plt.plot(angles,flap2/1000.,'--b')
        # plt.plot(angles,edge2/1000.,'-b')

        # plt.show()


        # T11
        #paths to the FAST output files
        filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
        filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
        # T5.6
        # filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C464_W8_T5.6_P0.0_m2D_L0/Model.out'
        # filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C437_W8_T5.6_P0.0_4D_L0/Model.out'
        # filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C455_W8_T5.6_P0.0_10D_L0/Model.out'


        """test edge setup"""
        # ffree, ofree, sfree, fclose, oclose, sclose, ffar, ofar, sfar = setup_atm_flap(filename_free,filename_close,filename_far)
        #
        # upper_free,lower_free,upper_close,lower_close,upper_far,lower_far = setup_edge_bounds(filename_free,filename_close,filename_far)
        #
        # m = make_edge_loads(upper_free,lower_free,ofree)
        #
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
        # filename = filename_free
        # lines = np.loadtxt(filename,skiprows=8)
        # time = lines[:,0]
        # mx_FAST = lines[:,11]
        #
        # plt.plot(m[5000:10000])
        # plt.plot(mx_FAST[5000:10000])
        # plt.show()



        """test flap setup"""
        # ffree, ofree, sfree, fclose, oclose, sclose, ffar, ofar, sfar = setup_atm_flap(filename_free,filename_close,filename_far)
        # T = np.linspace(0.,600.,24001)
        # ax1 = plt.subplot(131)
        # ax2 = plt.subplot(132)
        # ax3 = plt.subplot(133)
        #
        # ax1.plot(T,ffree)
        #
        # ax2.plot(T,fclose)
        #
        # ax3.plot(T,ffar)
        #
        # ax1.set_ylim(-3000.,3000.)
        # ax2.set_ylim(-3000.,3000.)
        # ax3.set_ylim(-3000.,3000.)
        #
        # plt.show()


        #
        #
        # plt.figure(1)
        # plt.plot(T,ffree,'-r')
        #
        # plt.figure(2)
        # plt.plot(T,efree,'-r')
        #
        # turbineX = np.array([0.,126.4])*4.
        # turbineY = np.array([0.,126.4])*0.5
        # turb_index = 1
        #
        # get_loads_history(turbineX,turbineY,turb_index,ofree,oclose,ofar,sfree,sclose,sfar,
        #                         ffree,fclose,ffar,efree,eclose,efar)
        #
        #
        # plt.show()
        # #

        # """test full"""
        # turbineX = np.array([0.,126.4])*7.
        # turbineY = np.array([0.,126.4])*0.5
        # turb_index = 1
        # s = Time.time()
        # ffree, ofree, sfree, fclose, oclose, sclose, ffar, ofar, sfar = setup_atm_flap(filename_free,filename_close,filename_far)
        # print 'setup flap: ', Time.time()-s
        # s = Time.time()
        # upper_free,lower_free,upper_close,lower_close,upper_far,lower_far = setup_edge_bounds(filename_free,filename_close,filename_far)
        # print 'setup edge: ', Time.time()-s
        # s = Time.time()
        # recovery_dist = find_freestream_recovery()
        # print 'setup recovery: ', Time.time()-s
        # s = Time.time()
        # flap,edge = get_loads_history(turbineX,turbineY,turb_index,ofree,oclose,ofar,sfree,sclose,
        #         sfar,ffree,fclose,ffar,upper_free,lower_free,upper_close,lower_close,
        #         upper_far,lower_far,recovery_dist)
        # print 'run loads: ', Time.time()-s
        #
        # # filename = filename_free
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C662_W8_T11.0_P0.0_7D_L0/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
        # lines = np.loadtxt(filename,skiprows=8)
        # mx_FAST = lines[:,11]
        # my_FAST = lines[:,12]
        #
        # plt.figure(1)
        # plt.plot(flap)
        # plt.plot(my_FAST)
        #
        # plt.figure(2)
        # plt.plot(edge)
        # plt.plot(mx_FAST)
        #
        # plt.show()
