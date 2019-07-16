import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C720_W8_T11.0_P0.5_m2D_L1.0/Model.out'
    filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C433_W8_T5.6_P0.0_4D_L-1.0/Model.out'
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
    MY2 = np.zeros_like(my)
    MY2[0] = my[0]
    n = len(time)
    for i in range(1,n-1):
        MY2[i] = (MY[i-1]+MY[i]+MY[i+1])/3.
    # print len(my)

    plt.ylim(0.,10000.)

    # plt.plot(time,my)
    plt.plot(time[0:1000],my[0:1000])
    plt.plot(time[0:1000],MY[0:1000])
    # plt.plot(time[0:1000],MY2[0:1000])
    # print time[0:1000]
    plt.show()
