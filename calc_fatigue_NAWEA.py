
import numpy as np
import math
from ccblade import *
from wakeLoadsFuncs import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as Time
import fast_calc_aep
import scipy.signal
from numpy import fabs as fabs
import damage_calc


"""preprocess"""
def extract_damage(filename_free,filename_close,filename_far):

    lines = np.loadtxt(filename_free,skiprows=8)
    atm_free = lines[:,12]

    lines = np.loadtxt(filename_close,skiprows=8)
    atm_close = lines[:,12]

    lines = np.loadtxt(filename_far,skiprows=8)
    atm_far = lines[:,12]

    damage_free = get_damage(atm_free,1.0,fos=3)
    damage_close = get_damage(atm_close,1.0,fos=3)
    damage_far = get_damage(atm_far,1.0,fos=3)

    return damage_free,damage_close,damage_far


"""preprocess"""
def get_damage(moments,freq,fos=3):

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
    I = 0.25*np.pi*(R**4-(R-0.08)**4)

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


"""preprocess"""
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


def farm_damage(turbineX,turbineY,windDirections,windFrequencies,damage_free,damage_close,damage_far,wind_speed=8,TI=0.11,rotor_diameter=126.4):
    """
    calculate the damage of each turbine in the farm for every wind direction

    inputs:
    turbineX:       x locations of the turbines (in the wind frame)
    turbineY:       y locations of the turbines (in the wind frame)
    windDirections: the wind directions
    windFrequencies: the associated probabilities of the wind directions
    atm_free:       the freestream atmospheric loads as a function of time
    atm_close:      the 4D downstream fully waked atmospheric loads as a function of time
    atm_far:        the 10D downstream fully waked atmospheric loads as a function of time

    outputs:
    damage:         fatigue damage of the farm

    """

    damage = np.zeros_like(turbineX)
    nDirections = len(windDirections)
    nTurbines = len(turbineX)

    for j in range(nDirections):
            turbineXw, turbineYw = fast_calc_aep.windframe(windDirections[j], turbineX, turbineY)
            ind = np.argsort(turbineXw)
            sortedX = np.zeros(nTurbines)
            sortedY = np.zeros(nTurbines)
            for k in range(nTurbines):
                sortedX[k] = turbineXw[ind[k]]
                sortedY[k] = turbineYw[ind[k]]
            # _, sigma = get_speeds(turbineXw, turbineYw, np.array([0.]), np.array([0.]), np.array([0.]), wind_speed, TI=TI)
            _, sigma = get_speeds(sortedX, sortedY, np.array([0.]), np.array([0.]), np.array([0.]), wind_speed, TI=TI)
            wake_radius = sigma*2.
            for i in range(nTurbines):
                    # damage[i] += damage_calc.combine_damage(turbineXw,turbineYw,i,damage_free,damage_close,damage_far,rotor_diameter,wake_radius)*windFrequencies[j]
                    damage[i] += damage_calc.combine_damage(sortedX,sortedY,np.where(ind==i)[0][0],damage_free,damage_close,damage_far,rotor_diameter,wake_radius)*windFrequencies[j]
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
