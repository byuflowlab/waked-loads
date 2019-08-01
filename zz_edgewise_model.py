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
import damage_calc




def find_freestream_recovery(TI=0.11,rotor_diameter=126.4,wind_speed=8.):
        turbineY = np.array([0.,0.])
        dx = np.linspace(0.,100.,1000)
        speeds = np.zeros(1000)
        for i in range(1000):
                turbineX = np.array([0.,126.4])*dx[i]
                speeds[i] = get_eff_turbine_speeds(turbineX, turbineY, wind_speed,TI=0.11)[1]
        func = interp1d(speeds,dx)
        return func(0.95*wind_speed)


def setup_edge(filename_free,filename_close,filename_far,TI=0.11,wind_speed=8.):
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
        edge_free = lines[:,11]
        Omega_free = np.mean(lines[:,6])

        """waked FAST CLOSE"""
        lines = np.loadtxt(filename_close,skiprows=8)
        edge_close = lines[:,11]
        Omega_close = np.mean(lines[:,6])

        """waked FAST FAR"""
        lines = np.loadtxt(filename_far,skiprows=8)
        edge_far = lines[:,11]
        Omega_far = np.mean(lines[:,6])

        N = len(edge_free)

        #find the peaks
        pp = scipy.signal.find_peaks(edge_free)[0]
        pn = scipy.signal.find_peaks(-edge_free)[0]
        upper = np.zeros(len(pp))
        lower = np.zeros(len(pn))
        for i in range(len(pp)):
                upper[i] = edge_free[pp[i]]
        for i in range(len(pn)):
                lower[i] = edge_free[pn[i]]

        index = np.array([])
        for i in range(len(upper)):
                if upper[i] < 0.:
                        index = np.append(index,i)
        upper = np.delete(upper,index)

        index = np.array([])
        for i in range(len(lower)):
                if lower[i] > 0.:
                        index = np.append(index,i)
        lower = np.delete(lower,index)

        upper_free = np.mean(upper)
        lower_free = np.mean(lower)


        pp = scipy.signal.find_peaks(edge_close)[0]
        pn = scipy.signal.find_peaks(-edge_close)[0]
        upper = np.zeros(len(pp))
        lower = np.zeros(len(pn))
        for i in range(len(pp)):
                upper[i] = edge_close[pp[i]]
        for i in range(len(pn)):
                lower[i] = edge_close[pn[i]]

        index = np.array([])
        for i in range(len(upper)):
                if upper[i] < 0.:
                        index = np.append(index,i)
        upper = np.delete(upper,index)

        index = np.array([])
        for i in range(len(lower)):
                if lower[i] > 0.:
                        index = np.append(index,i)
        lower = np.delete(lower,index)

        upper_close = np.mean(upper)
        lower_close = np.mean(lower)


        pp = scipy.signal.find_peaks(edge_far)[0]
        pn = scipy.signal.find_peaks(-edge_far)[0]
        upper = np.zeros(len(pp))
        lower = np.zeros(len(pn))
        for i in range(len(pp)):
                upper[i] = edge_far[pp[i]]
        for i in range(len(pn)):
                lower[i] = edge_far[pn[i]]

        index = np.array([])
        for i in range(len(upper)):
                if upper[i] < 0.:
                        index = np.append(index,i)
        upper = np.delete(upper,index)

        index = np.array([])
        for i in range(len(lower)):
                if lower[i] > 0.:
                        index = np.append(index,i)
        lower = np.delete(lower,index)

        upper_far = np.mean(upper)
        lower_far = np.mean(lower)

        turbineX_close = np.array([0.,126.4])*4.
        turbineX_far = np.array([0.,126.4])*10.

        turbineY_waked = np.array([0.,0.])

        free_speed = wind_speed
        close_speed = get_eff_turbine_speeds(turbineX_close, turbineY_waked, wind_speed,TI=TI)[1]
        far_speed = get_eff_turbine_speeds(turbineX_far, turbineY_waked, wind_speed,TI=TI)[1]


        return upper_free, lower_free, Omega_free, free_speed, upper_close, lower_close, Omega_close, close_speed, upper_far, lower_far, Omega_far, far_speed


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


def make_edge_loads(upper,lower,Omega,N=24001,duration=10.):
        n_cycles = Omega*duration
        t = np.linspace(0.,float(n_cycles),N)
        amp = (upper-lower)/2.
        avg = (upper+lower)/2.
        m = np.sin(t*2.*np.pi)*amp+avg
        return m


def get_edge_loads(turbineX,turbineY,turb_index,upper_free,lower_free,Omega_free,free_speed,upper_close,lower_close,Omega_close,close_speed,upper_far,lower_far,Omega_far,far_speed,recovery_dist,wind_speed=8.0,TI=0.11):

        nTurbines = len(turbineX)

        hub_height = 90.

        o = np.array([Omega_free,Omega_far,Omega_close,Omega_close])
        sp = np.array([free_speed,far_speed,close_speed,0.])
        f_o = interp1d(sp,o,kind='linear')

        actual_speed = get_eff_turbine_speeds(turbineX, turbineY, wind_speed,TI=TI)[turb_index]
        Omega = f_o(actual_speed)

        """amount waked"""
        _, sigma = get_speeds(turbineX, turbineY, np.array([0.]), np.array([0.]), np.array([0.]), wind_speed, TI=TI)
        wake_radius = sigma*2.

        waked_amount = np.zeros(nTurbines)
        dx_dist = np.zeros(nTurbines)

        amnt_waked = np.zeros(nTurbines)
        for waking in range(nTurbines):
                dx_dist[waking] = turbineX[turb_index]-turbineX[waking]
                dy = turbineY[turb_index]-turbineY[waking]
                amnt_waked[waking] = rotor_amount_waked(dy,wake_radius[turb_index][waking],rotor_diameter)

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

        # see how to interpolate the FAST data
        d_recover = recovery_dist-10.

        print 'amnt_waked: ', amnt_waked

        # upper = 0.
        # lower = 0.
        # for k in range(np.count_nonzero(waked_array)):
        #         if down[k] <= 4.:
        #                 upper += upper_close*waked_array[k]
        #                 lower += lower_close*waked_array[k]
        #         elif down[k] >= recovery_dist:
        #                 upper += upper_free*waked_array[k]
        #                 lower += lower_free*waked_array[k]
        #         elif down[k] >= 10. and down[k] < recovery_dist:
        #                 upper += (upper_far*(recovery_dist-down[k])/d_recover+upper_free*(down[k]-10.)/d_recover)*waked_array[k]
        #                 lower += (lower_far*(recovery_dist-down[k])/d_recover+lower_free*(down[k]-10.)/d_recover)*waked_array[k]
        #         else:
        #                 upper += (upper_close*(10.-down[k])/6.+upper_far*(down[k]-4.)/6.)*waked_array[k]
        #                 lower += (lower_close*(10.-down[k])/6.+lower_far*(down[k]-4.)/6.)*waked_array[k]
        #
        # upper += upper_free*unwaked
        # lower += lower_free*unwaked

        upper = 0.
        lower = 0.
        for k in range(np.count_nonzero(waked_array)):
                if down[k] <= 4.:
                        # upper += upper_close*waked_array[k]
                        lower += lower_close*waked_array[k]
                elif down[k] >= recovery_dist:
                        # upper += upper_free*waked_array[k]
                        lower += lower_free*waked_array[k]
                elif down[k] >= 10. and down[k] < recovery_dist:
                        # upper += (upper_far*(recovery_dist-down[k])/d_recover+upper_free*(down[k]-10.)/d_recover)*waked_array[k]
                        lower += (lower_far*(recovery_dist-down[k])/d_recover+lower_free*(down[k]-10.)/d_recover)*waked_array[k]
                else:
                        # upper += (upper_close*(10.-down[k])/6.+upper_far*(down[k]-4.)/6.)*waked_array[k]
                        lower += (lower_close*(10.-down[k])/6.+lower_far*(down[k]-4.)/6.)*waked_array[k]

        upper += upper_free
        lower += lower_free*unwaked

        # EDGEWISE MOMENTS
        moments_edge = make_edge_loads(upper,lower,Omega)

        return moments_edge






def calc_damage_moments(m_flap,m_edge,freq,fos=2):

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
    # I = 0.25*np.pi*R**4
    I = 0.25*np.pi*(R**4-(R-0.08)**4)

    #go from moments to stresses
    # sigma_flap = m_flap*R/I
    sigma_edge = m_edge*R/I
    # sigma = np.sqrt(sigma_flap**2+sigma_edge**2)

    sigma = sigma_edge

    # resultant_moment = np.sqrt(m_flap**2+m_edge**2)
    # sigma = resultant_moment*R/I

    # sigma = m_edge*R/I

    # plt.plot(sigma,linewidth=0.5)
    # plt.ylim(0.,60000.)
    # plt.show()

    #find the peak stresses
    pp = scipy.signal.find_peaks(sigma)[0]
    pn = scipy.signal.find_peaks(-sigma)[0]
    p = np.append(pn,pp)
    # p = np.append(0,pp)
    # p = np.append(p,pn)
    # p = np.append(p,len(sigma)-1)
    p = np.sort(p)
    peaks = np.zeros(len(p))
    # vv = np.arange(len(sigma))
    # v = np.zeros(len(p))
    for i in range(len(p)):
        peaks[i] = sigma[p[i]]
        # v[i] = vv[p[i]]

    # plt.plot(vv,sigma)
    # plt.plot(v,peaks,'o')
    # plt.xlim(0.,1000.)
    # plt.ylim(0.,120000.)
    # plt.title('flapwise')
    # plt.show()


    print len(peaks)

    #rainflow counting
    array = rainflow(peaks)

    alternate = array[0,:]/2.
    mean = array[1,:]
    count = array[3,:]

    # Goodman correction
    # su = 345000.
    su = 459000.
    # mar = alternate/(1.-mean/su)
    mar = alternate*(su/(su-mean))

    # mar = mar*1.3

    npts = len(mar)

    #damage calculations

    m = 10.
    for i in range(npts):
        # Nfail = (su/mar[i])**m
        # Nfail = 10.**((-mar[i]/su+1.)/0.1)
        Nfail = ((su)/(mar[i]))**m
        mult = 20.*365.*24.*6.*freq
        d += count[i]*mult/Nfail
    #     plt.plot(i,d,'o')
    # plt.show()

    fos = 1.
    return d*fos



#
#
def loads_farm_damage(turbineX,turbineY,windDirections,windFrequencies,atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed,TI=0.11,N=24001):
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


#
#
#
def preprocess_damage(filename_free,filename_close,filename_far,TI=0.11,rotor_diameter=126.4,wind_speed=8.):

        """free FAST"""
        lines = np.loadtxt(filename_free,skiprows=8)
        flap_free = lines[:,12]

        """waked FAST CLOSE"""
        lines = np.loadtxt(filename_close,skiprows=8)
        flap_close = lines[:,12]

        """waked FAST FAR"""
        lines = np.loadtxt(filename_far,skiprows=8)
        flap_far = lines[:,12]

        m_edge = np.zeros_like(flap_far)

        damage_free = calc_damage_moments(flap_free,m_edge,1.0)
        damage_close = calc_damage_moments(flap_close,m_edge,1.0)
        damage_far = calc_damage_moments(flap_far,m_edge,1.0)

        recovery_dist = find_freestream_recovery(TI=TI,rotor_diameter=rotor_diameter,wind_speed=wind_speed)

        return damage_free, damage_close, damage_far, recovery_dist
#
#
def damage_farm_damage(turbineX,turbineY,windDirections,windFrequencies,damage_free,damage_close,damage_far,recovery_dist,wind_speed=8,TI=0.11,rotor_diameter=126.4):
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
            _, sigma = get_speeds(sortedX, sortedY, np.array([0.]), np.array([0.]), np.array([0.]), wind_speed, TI=TI)
            wake_radius = sigma*2.
            for i in range(nTurbines):
                    # damage[i] += damage_calc.combine_damage(turbineXw,turbineYw,i,damage_free,damage_close,damage_far,rotor_diameter,wake_radius)*windFrequencies[j]
                    damage[i] += damage_calc.combine_damage(sortedX,sortedY,np.where(ind==i)[0][0],damage_free,damage_close,damage_far,rotor_diameter,wake_radius,recovery_dist)*windFrequencies[j]
    return damage



if __name__ == '__main__':

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
        rotor_diameter = 126.4
        turbineX = np.array([0.,4.])*rotor_diameter
        turbineY = np.array([0.,-0.5])*rotor_diameter
        turb_index = 1

        s = Time.time()
        recovery_dist = find_freestream_recovery()
        print 'recovery distance time: ', Time.time()-s

        s = Time.time()
        upper_free,lower_free,Omega_free,free_speed,upper_close,lower_close,Omega_close,close_speed,upper_far,lower_far,Omega_far,far_speed = setup_edge(filename_free,filename_close,filename_far)
        print 'edge setup time: ', Time.time()-s

        s = Time.time()
        moments = get_edge_loads(turbineX,turbineY,turb_index,upper_free,lower_free,Omega_free,free_speed,upper_close,lower_close,Omega_close,close_speed,upper_far,lower_far,Omega_far,far_speed,recovery_dist)
        print 'make loads time: ', Time.time()-s


        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C651_W8_T11.0_P0.0_4D_L-0.5/Model.out'
        #
        # lines = np.loadtxt(filename,skiprows=8)
        # edge_close = lines[:,11]
        #
        # plt.plot(moments)
        # plt.plot(edge_close)
        # plt.show()


        sep = np.array([4.,7.,10.])
        offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
        # sep = np.array([4.])
        # offset = np.array([0.5])
        TI = 11.0
        for i in range(len(sep)):
                damage = np.zeros_like(offset)
                for j in range(len(offset)):
                        turbineX = np.array([0.,1.])*rotor_diameter*sep[i]
                        turbineY = np.array([0.,1.])*rotor_diameter*offset[j]

                        moments = get_edge_loads(turbineX,turbineY,turb_index,upper_free,lower_free,Omega_free,free_speed,upper_close,lower_close,Omega_close,close_speed,upper_far,lower_far,Omega_far,far_speed,recovery_dist)
                        my = 0.
                        damage[j] = calc_damage_moments(my,moments,1.0,fos=2.)
                        # damage[j] = get_damage(my,1.0,fos=2.0)
                        # new_calc_damage(my,mx,1.0)
                        print damage[j]
                        # plt.figure(1)
                        # plt.plot(offset[j],1.2885465608898776,'or')

                plt.figure(1)
                plt.plot(offset,damage,'o')
                print 'separation: ', sep[i]
                print 'damage: ', repr(damage)
        plt.show()
