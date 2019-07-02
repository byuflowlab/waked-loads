import numpy as np
import matplotlib.pyplot as plt
import rainflow


def damage_soderberg(m,R):
    d = 0.
    I = 0.25*np.pi*R**4
    y = R
    sigma = m*y/I

    print max(sigma)
    c = rainflow.extract_cycles(sigma)

    mean = np.array([])
    alternate = np.array([])
    m = np.array([])
    for low, high, mult in c:
        mean = np.append(mean,0.5 * (high + low))
        alternate = np.append(alternate,(high-low)/2.)
        m = np.append(m,mult)

    Smax = 80.0*1.E6
    ex = 4.
    sig_eff = alternate*(Smax/(Smax-mean))
    # sig_eff = (1.-mean*Smax)/alternate
    Nf = (Smax/sig_eff)**ex

    for i in range(len(mean)):
        d += m[i]/Nf[i]

    print d*52560.0*20.


if __name__ == '__main__':

    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C720_W8_T11.0_P0.5_m2D_L1.0/Model.out'
    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C682_W8_T11.0_P0.0_m2D_L0.5/Model.out'
    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C649_W8_T11.0_P0.0_4D_L-1.0/Model.out'
    lines = np.loadtxt(filename,skiprows=8)
    # print np.shape(lines)
    time = lines[:,0]
    mx = lines[:,11]
    my = lines[:,12]

    MY = np.zeros_like(my)
    MY[0] = my[0]
    n = len(time)
    for i in range(1,n-1):
        MY[i] = (my[i-1]+my[i]+my[i+1])/3.
    MY[-1] = my[-1]

    MY2 = np.zeros_like(my)
    MY2[0] = MY[0]
    n = len(time)
    for i in range(1,n-1):
        MY2[i] = (MY[i-1]+MY[i]+MY[i+1])/3.
    MY2[-1] = my[-1]
    print 3600.*24.*365/(time[-1]-time[0])

    # cycles = rainflow.count_cycles(my[0:1000])
    # c = rainflow.extract_cycles(my[0:1000])
    #
    # mean = np.array([])
    # alternate = np.array([])
    # t = 0
    # for low, high, mult in c:
    #     mean = np.append(mean,0.5 * (high + low))
    #     alternate = np.append(alternate,(high-low)/2.)
    #     t += mult
    # print t
    # print mean
    # print alternate
    # print cycles
    #
    # print np.shape(mean)
    # print np.shape(alternate)
    # print np.shape(cycles)
    #
    # n = np.shape(cycles)[0]
    # t = 0
    # for i in range(n):
    #     t += cycles[i][1]
    # print t

    # print low
    # print high
    # print mean
    # print rng
    # print mult
    # c.extract_cycles()

    damage_soderberg(my,0.5)
    damage_soderberg(MY,0.5)
    damage_soderberg(MY2,0.5)
    # print len(my)

    # plt.ylim(0.,10000.)

    # plt.plot(time[0:1000],my[0:1000])
    # plt.plot(time[0:100],my[0:100])
    # plt.plot(time[0:1000],MY2[0:1000])
    # print time[0:1000]
    # plt.show()
