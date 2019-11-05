import numpy as np
import math
from ccblade import *
from wakeLoadsFuncs import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as Time
import scipy.signal

# import damage_calc
from numpy import fabs as fabs
import numpy as np


def WindFrame(wind_direction, turbineX, turbineY):
    """ Calculates the locations of each turbine in the wind direction reference frame """
    nTurbines = len(turbineX)
    windDirectionDeg = wind_direction
    # adjust directions
    windDirectionDeg = 270. - windDirectionDeg
    if windDirectionDeg < 0.:
        windDirectionDeg += 360.
    windDirectionRad = np.pi*windDirectionDeg/180.0    # inflow wind direction in radians

    # convert to downwind(x)-crosswind(y) coordinates
    turbineXw = turbineX*np.cos(-windDirectionRad)-turbineY*np.sin(-windDirectionRad)
    turbineYw = turbineX*np.sin(-windDirectionRad)+turbineY*np.cos(-windDirectionRad)

    return turbineXw, turbineYw


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



def get_edgewise_damage(turbineX,turbineY,turb_index,Rhub,r,chord,theta,af,Rtip,B,rho,mu,
                        precone,hubHt,nSector,pitch,yaw_deg,TI=0.11):

        free_speed = 8.
        actual_speed = get_eff_turbine_speeds(turbineX, turbineY, free_speed,TI=TI)[turb_index]
        TSR = 7.55
        Omega = actual_speed*TSR/Rtip/(2.*np.pi)*60.
        print Omega

        az = 90.
        x_locs,y_locs,z_locs = findXYZ(turbineX[turb_index],turbineY[turb_index],hubHt,r,yaw_deg,az)
        speeds, _ = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, free_speed, TI=TI)
        _,edge90 = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az)

        az = 270.
        x_locs,y_locs,z_locs = findXYZ(turbineX[turb_index],turbineY[turb_index],hubHt,r,yaw_deg,az)
        speeds, _ = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, free_speed, TI=TI)
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


def farm_damage(turbineX,turbineY,windDirections,windFrequencies,Rhub,r,chord,theta,af,
                        Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=0.11):

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
            turbineXw, turbineYw = WindFrame(windDirections[j], turbineX, turbineY)
            for i in range(nTurbines):
                damage[i] += get_edgewise_damage(turbineXw,turbineYw,i,Rhub,r,chord,theta,
                                af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI) * windFrequencies[j]
    return damage


if __name__ == '__main__':

        turbineX = np.array([0.,500.])
        turbineY = np.array([0.,100.])
        windDirections = np.array([270.])
        windFrequencies = np.array([1.])

        farm_damage(turbineX,turbineY,windDirections,windFrequencies,Rhub,r,chord,theta,af,
                                Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=0.11)
