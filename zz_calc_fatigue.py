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




def find_freestream_recovery(TI=0.11,rotor_diameter=126.4,wind_speed=8.):
        turbineY = np.array([0.,0.])
        dx = np.linspace(0.,100.,1000)
        speeds = np.zeros(1000)
        for i in range(1000):
                turbineX = np.array([0.,126.4])*dx[i]
                speeds[i] = get_eff_turbine_speeds(turbineX, turbineY, wind_speed,TI=0.11)[1]
        func = interp1d(speeds,dx)
        return func(0.95*wind_speed)


def setup_atm_flap(filename_free,filename_close,filename_far,TI=0.11,wind_speed=8.):
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
        flap_free = lines[:,12]
        # edge_free = lines[:,11]
        Omega_free = np.mean(lines[:,6])

        """waked FAST CLOSE"""
        lines = np.loadtxt(filename_close,skiprows=8)
        time = lines[:,0]
        flap_close = lines[:,12]
        # edge_close = lines[:,11]
        Omega_close = np.mean(lines[:,6])

        # plt.figure(1)
        # plt.plot(edge_close)

        """waked FAST FAR"""
        lines = np.loadtxt(filename_far,skiprows=8)
        time = lines[:,0]
        flap_far = lines[:,12]
        # edge_far = lines[:,11]
        Omega_far = np.mean(lines[:,6])


        """setup the CCBlade loads"""
        turbineX_close = np.array([0.,126.4])*4.
        turbineX_far = np.array([0.,126.4])*10.

        turbineY_free = np.array([0.,1000000.])
        turbineY_waked = np.array([0.,0.])

        Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()

        hub_height = 90.

        angles = np.linspace(0.,360.,2)
        flap_free_cc = np.zeros_like(angles)
        # edge_free_cc = np.zeros_like(angles)
        flap_close_cc = np.zeros_like(angles)
        # edge_close_cc = np.zeros_like(angles)
        flap_far_cc = np.zeros_like(angles)
        # edge_far_cc = np.zeros_like(angles)

        for i in range(len(angles)):
                az = angles[i]
                #freestream
                x_locs,y_locs,z_locs = findXYZ(turbineX_close[1],turbineY_free[1],hub_height,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX_close, turbineY_free, x_locs, y_locs, z_locs, wind_speed,TI=TI)
                # flap_free_cc[i], edge_free_cc[i] = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_free,pitch,azimuth=az)
                flap_free_cc[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_free,pitch,azimuth=az)

                #waked close
                x_locs,y_locs,z_locs = findXYZ(turbineX_close[1],turbineY_waked[1],hub_height,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX_close, turbineY_waked, x_locs, y_locs, z_locs, wind_speed,TI=TI)
                # flap_close_cc[i], edge_close_cc[i] = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_close,pitch,azimuth=az)
                flap_close_cc[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_close,pitch,azimuth=az)

                #waked far
                x_locs,y_locs,z_locs = findXYZ(turbineX_far[1],turbineY_waked[1],hub_height,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX_far, turbineY_waked, x_locs, y_locs, z_locs, wind_speed,TI=TI)
                # flap_far_cc[i], edge_far_cc[i] = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_far,pitch,azimuth=az)
                flap_far_cc[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega_far,pitch,azimuth=az)

        free_speed = wind_speed
        close_speed = get_eff_turbine_speeds(turbineX_close, turbineY_waked, wind_speed,TI=TI)[1]
        far_speed = get_eff_turbine_speeds(turbineX_far, turbineY_waked, wind_speed,TI=TI)[1]

        N = len(flap_free)
        pos_free = np.linspace(0.,Omega_free*10.*360.,N)%360.
        pos_close = np.linspace(0.,Omega_close*10.*360.,N)%360.
        pos_far = np.linspace(0.,Omega_far*10.*360.,N)%360.

        f_free_flap = interp1d(angles,flap_free_cc)
        # f_free_edge = interp1d(angles,edge_free_cc)
        f_close_flap = interp1d(angles,flap_close_cc)
        # f_close_edge = interp1d(angles,edge_close_cc)
        f_far_flap = interp1d(angles,flap_far_cc)
        # f_far_edge = interp1d(angles,edge_far_cc)

        """get atm loads"""
        flap_free_atm = flap_free-f_free_flap(pos_free)/1000.
        # edge_free_atm = edge_free-f_free_edge(pos_free)/1000.

        flap_close_atm = flap_close-f_close_flap(pos_close)/1000.
        # edge_close_atm = edge_close-f_close_edge(pos_close)/1000.

        flap_far_atm = flap_far-f_far_flap(pos_far)/1000.
        # edge_far_atm = edge_far-f_far_edge(pos_far)/1000.


        # return flap_free_atm, edge_free_atm, Omega_free, free_speed, flap_close_atm, edge_close_atm, Omega_close, close_speed, flap_far_atm, edge_far_atm, Omega_far, far_speed
        return flap_free_atm, Omega_free, free_speed, flap_close_atm, Omega_close, close_speed, flap_far_atm, Omega_far, far_speed


def setup_edge_bounds(filename_free,filename_close,filename_far,TI=0.11,wind_speed=8.):
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

        """waked FAST CLOSE"""
        lines = np.loadtxt(filename_close,skiprows=8)
        edge_close = lines[:,11]

        """waked FAST FAR"""
        lines = np.loadtxt(filename_far,skiprows=8)
        edge_far = lines[:,11]

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


        return upper_free, lower_free, upper_close, lower_close, upper_far, lower_far


def make_edge_loads(upper,lower,Omega,N=24001,duration=10.):
        n_cycles = Omega*duration
        t = np.linspace(0.,float(n_cycles),N)
        amp = (upper-lower)/2.
        avg = (upper+lower)/2.
        m = np.sin(t*2.*np.pi)*amp+avg
        return m





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
    # sigma_edge = m_edge*R/I
    # sigma = np.sqrt(sigma_flap**2+sigma_edge**2)

    resultant_moment = np.sqrt(m_flap**2+m_edge**2)
    sigma = resultant_moment*R/I

    # plt.plot(sigma,linewidth=0.5)
    # plt.ylim(0.,60000.)
    # plt.show()

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

    plt.plot(vv,sigma)
    plt.plot(v,peaks,'o')
    plt.xlim(0.,1000.)
    plt.ylim(0.,120000.)
    plt.title('flapwise')
    plt.show()


    print len(peaks)

    #rainflow counting
    array = rainflow(peaks)

    alternate = array[0,:]/2.
    mean = array[1,:]
    count = array[3,:]

    # Goodman correction
    su = 345000.
    # mar = alternate/(1.-mean/su)
    mar = alternate*(su/(su-mean))

    npts = len(mar)

    #damage calculations

    m = 10.
    for i in range(npts):
        # Nfail = (su/mar[i])**m
        # Nfail = 10.**((-mar[i]/su+1.)/0.1)
        Nfail = ((su)/(mar[i]))**m
        mult = 25.*365.*24.*6.*freq
        d += count[i]*mult/Nfail

    return d*fos










def new_calc_damage(m_flap,m_edge,freq,fos=3):

    """
    calculate the damage of a turbine from a single direction

    inputs:
    moments:        the moment history
    freq:           the probability of these moments
    fos:            factor of safety


    outputs:
    damage:         fatigue damage

    """

    EI = 18110.E6
    m = 10.


    # plt.plot(sigma,linewidth=0.5)
    # plt.ylim(0.,60000.)
    # plt.show()

    """FLAPWISE"""
    plt.plot(m_flap)
    plt.plot(m_edge)
    plt.show()
    pp = scipy.signal.find_peaks(m_flap)[0]
    pn = scipy.signal.find_peaks(-m_flap)[0]
    p = np.append(0,pp)
    p = np.append(p,pn)
    p = np.append(p,len(m_flap)-1)
    p = np.sort(p)
    peaks = np.zeros(len(p))
    for i in range(len(p)):
        peaks[i] = m_flap[p[i]]
    #rainflow counting
    array = rainflow(peaks)

    L_range = array[0,:]
    mean = array[1,:]
    count = array[3,:]
    nCycles = np.sum(count)

    # Goodman correction
    su = 345000.
    corrected = L_range/(1.-mean/su)
    npts = len(corrected)


    summation = 0.
    for i in range(npts):
            summation += count[i]*corrected[i]**m

    DEM_flap = (summation/nCycles)**(1./m)



    """EDGEWISE"""
    pp = scipy.signal.find_peaks(m_edge)[0]
    pn = scipy.signal.find_peaks(-m_edge)[0]
    p = np.append(0,pp)
    p = np.append(p,pn)
    p = np.append(p,len(m_edge)-1)
    p = np.sort(p)
    peaks = np.zeros(len(p))
    for i in range(len(p)):
        peaks[i] = m_edge[p[i]]
    #rainflow counting
    array = rainflow(peaks)

    L_range = array[0,:]
    mean = array[1,:]
    count = array[3,:]
    nCycles = np.sum(count)

    # Goodman correction
    su = 345000.
    corrected = L_range/(1.-mean/su)
    npts = len(corrected)

    summation = 0.
    for i in range(npts):
            summation += count[i]*corrected[i]**m

    DEM_edge = (summation/nCycles)**(1./m)


    eps = -0.5*(DEM_flap/EI-DEM_edge/EI)
    eps_max = 0.005
    Nfail = (eps_max/(fos*eps))**m

    mult = 25.*365.*24.*6.*freq
    print Nfail
    # print DEM_flap
    # print DEM_flap/EI
    # print 'EI: ', EI
    # print 'eps: ', eps
    #
    #
    # print 'flap: ', DEM_flap
    # print 'edge: ', DEM_edge










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


def get_loads_history(turbineX,turbineY,turb_index,Omega_free,Omega_close,Omega_far,free_speed,close_speed,
        far_speed,flap_atm_free,flap_atm_close,flap_atm_far,upper_free,lower_free,upper_close,lower_close,
        upper_far,lower_far,recovery_dist,TI=0.11,wind_speed=8.,rotor_diameter=126.4):

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
        N = len(flap_atm_close)

        Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()
        # angles = np.linspace(0.,360.,100)
        angles = np.linspace(0.,360.,50)
        flap_ccblade_moments = np.zeros_like(angles)

        hub_height = 90.

        o = np.array([Omega_free,Omega_far,Omega_close,Omega_close])
        sp = np.array([free_speed,far_speed,close_speed,0.])
        f_o = interp1d(sp,o,kind='linear')

        s = Time.time()
        """CCBlade moments"""
        for i in range(len(angles)):
                az = angles[i]
                x_locs,y_locs,z_locs = findXYZ(turbineX[turb_index],turbineY[turb_index],hub_height,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, wind_speed,TI=TI)

                if i == 0:
                    actual_speed = get_eff_turbine_speeds(turbineX, turbineY, wind_speed,TI=TI)[turb_index]
                    Omega = f_o(actual_speed)

                flap_ccblade_moments[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az)

        f_flap = interp1d(angles, flap_ccblade_moments/1000., kind='cubic')
        pos = np.linspace(0.,Omega*10.*360.,N)%360.
        print 'CCBlade time: ', Time.time()-s


        s = Time.time()
        """amount waked"""
        _, sigma = get_speeds(turbineX, turbineY, np.array([0.]), np.array([0.]), np.array([0.]), wind_speed, TI=TI)
        wake_radius = sigma*2.6

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

        print 'waking amount time: ', Time.time()-s

        s = Time.time()
        # see how to interpolate the FAST data
        d_recover = recovery_dist-10.
        moments_flap = f_flap(pos)
        upper = 0.
        lower = 0.
        for k in range(np.count_nonzero(waked_array)):
                if down[k] <= 4.:
                        moments_flap += flap_atm_close*waked_array[k]
                        upper += upper_close*waked_array[k]
                        lower += lower_close*waked_array[k]
                elif down[k] >= recovery_dist:
                        moments_flap += flap_atm_free*waked_array[k]
                        upper += upper_free*waked_array[k]
                        lower += lower_free*waked_array[k]
                elif down[k] >= 10. and down[k] < recovery_dist:
                        moments_flap += (flap_atm_far*(recovery_dist-down[k])/d_recover+flap_atm_free*(down[k]-10.)/d_recover)*waked_array[k]
                        upper += (upper_far*(recovery_dist-down[k])/d_recover+upper_free*(down[k]-10.)/d_recover)*waked_array[k]
                        lower += (lower_far*(recovery_dist-down[k])/d_recover+lower_free*(down[k]-10.)/d_recover)*waked_array[k]
                else:
                        moments_flap += (flap_atm_close*(10.-down[k])/6.+flap_atm_far*(down[k]-4.)/6.)*waked_array[k]
                        upper += (upper_close*(10.-down[k])/6.+upper_far*(down[k]-4.)/6.)*waked_array[k]
                        lower += (lower_close*(10.-down[k])/6.+lower_far*(down[k]-4.)/6.)*waked_array[k]
        moments_flap += flap_atm_free*unwaked
        upper += upper_free*unwaked
        lower += lower_free*unwaked

        # EDGEWISE MOMENTS
        moments_edge = make_edge_loads(upper,lower,Omega)

        print 'combine time: ', Time.time()-s

        return moments_flap, moments_edge


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

        """test full"""
        turbineX = np.array([0.,126.4])*7.
        turbineY = np.array([0.,126.4])*0.5
        turb_index = 1
        s = Time.time()
        ffree, ofree, sfree, fclose, oclose, sclose, ffar, ofar, sfar = setup_atm_flap(filename_free,filename_close,filename_far)
        print 'setup flap: ', Time.time()-s
        s = Time.time()
        upper_free,lower_free,upper_close,lower_close,upper_far,lower_far = setup_edge_bounds(filename_free,filename_close,filename_far)
        print 'setup edge: ', Time.time()-s
        s = Time.time()
        recovery_dist = find_freestream_recovery()
        print 'setup recovery: ', Time.time()-s
        s = Time.time()
        flap,edge = get_loads_history(turbineX,turbineY,turb_index,ofree,oclose,ofar,sfree,sclose,
                sfar,ffree,fclose,ffar,upper_free,lower_free,upper_close,lower_close,
                upper_far,lower_far,recovery_dist)
        print 'run loads: ', Time.time()-s

        # filename = filename_free
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C662_W8_T11.0_P0.0_7D_L0/Model.out'
        filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
        lines = np.loadtxt(filename,skiprows=8)
        mx_FAST = lines[:,11]
        my_FAST = lines[:,12]

        plt.figure(1)
        plt.plot(flap)
        plt.plot(my_FAST)

        plt.figure(2)
        plt.plot(edge)
        plt.plot(mx_FAST)

        plt.show()
