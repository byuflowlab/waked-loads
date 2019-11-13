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


def get_edgewise_damage(turbineX,turbineY,turb_index,TSR_TSR,TSR_speeds,free_speed,Rhub,r,chord,theta,
                        af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=0.11):

        actual_speed = get_eff_turbine_speeds(turbineX, turbineY, free_speed,TI=TI)[turb_index]
        if isinstance(TSR_TSR,float):
                TSR = TSR_TSR
        else:
                f_TSR = interp1d(TSR_speeds, TSR_TSR, kind='cubic')
                TSR = f_TSR(actual_speed)


        Omega = TSR*actual_speed/Rtip
        Omega = Omega*9.5492965964254 #rad/s to RPM

        az = 90.
        x_locs,y_locs,z_locs = findXYZ(turbineX[turb_index],turbineY[turb_index],hubHt,r,yaw_deg,az)

        speeds, _ = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, free_speed, TI=TI)
        # edge90 = calc_moment_edge(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,wind_speed,pitch,azimuth=az)
        _,edge90 = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az)
        # print 'edge90: ', edge90
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
        su = 535000.
        effective = alternate/(1.-mean/su)

        m = 10.
        fos = 1.15
        Nfail = (su/(effective*fos))**m #mLife

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


def farm_damage(turbineX,turbineY,windDirections,windFrequencies,windSpeeds,TSR_TSR,TSR_speeds,
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
                damage[i] += get_edgewise_damage(turbineXw,turbineYw,i,TSR_TSR,TSR_speeds,windSpeeds[j],
                                        Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI)*windFrequencies[j]

    return damage


if __name__ == '__main__':

        windDirections = np.array([270.])
        windFrequencies = np.array([1.])
        windSpeeds = np.array([8.])

        TSR_speeds = np.array([0.,3.1393,3.6744,4.0369,5.0215,6.0238,7.0089,7.8040,9.0142,
                        10.017,11.382,12.022,13.007,14.010,14.995,15.998,17.000,
                        18.020,19.023,19.991,20.993,21.996,22.999,24.001,24.969,100.])
        TSR_TSR = np.array([14.848,14.848,12.945,11.648,9.7426,8.6149,7.8329,7.3973,7.3918,
                        7.3872,6.8627,6.5142,6.0777,5.4684,5.1184,4.7683,4.5045,
                        4.3271,4.0634,3.8862,3.6224,3.4451,3.2677,3.1768,2.9996,2.9996])

        Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()

        spac = np.linspace(-3.,3.,25)
        print spac
        D = Rtip*2.
        dam = np.zeros(25)

        for i in range(25):
                turbineX = np.array([0.,D*10.])
                turbineY = np.array([0.,spac[i]*D])

                dam[i] =  farm_damage(turbineX,turbineY,windDirections,windFrequencies,windSpeeds,TSR_TSR,TSR_speeds,
                                        Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=0.056)[1]

        print repr(dam)
