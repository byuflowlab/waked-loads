import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl



if __name__=="__main__":

    fig = plt.figure(1,figsize=[6.,6.])
    label_size = 8
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    fontProperties = {'family':'serif','size':8}

    filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
    filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
    filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'

    lines = np.loadtxt(filename_free,skiprows=8)
    t = lines[:,0]
    t = t-t[0]
    loads_free = lines[:,12]

    lines = np.loadtxt(filename_close,skiprows=8)
    loads_close = lines[:,12]

    lines = np.loadtxt(filename_far,skiprows=8)
    loads_far = lines[:,12]

    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

    ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
    ax1.set_yticklabels(ax1.get_yticks(), fontProperties)
    ax2.set_xticklabels(ax2.get_xticks(), fontProperties)
    ax2.set_yticklabels(ax2.get_yticks(), fontProperties)
    ax3.set_xticklabels(ax1.get_xticks(), fontProperties)
    ax3.set_yticklabels(ax1.get_yticks(), fontProperties)

    ax1.plot(t,loads_free,color='C0',linewidth=0.5)
    ax1.set_xticks((0.,300.,600.))
    ax1.set_xticklabels(('','',''))
    ax1.set_yticks((2000.,4500.,7000.))
    ax1.set_yticklabels(('2000','4500','7000'))
    ax1.set_xlim(0.,600.)
    ax1.set_ylim(2000.,7000.)
    ax1.grid()

    ax2.plot(t,loads_close,color='C0',linewidth=0.5)
    ax2.set_xticks((0.,300.,600.))
    ax2.set_xticklabels(('','',''))
    ax2.set_yticks((2000.,4500.,7000.))
    ax2.set_yticklabels(('2000','4500','7000'))
    ax2.set_xlim(0.,600.)
    ax2.set_ylim(2000.,7000.)
    ax2.grid()

    ax3.plot(t,loads_far,color='C0',linewidth=0.5)
    ax3.set_xticks((0.,300.,600.))
    ax3.set_xticklabels(('0','300','600'))
    ax3.set_yticks((2000.,4500.,7000.))
    ax3.set_yticklabels(('2000','4500','7000'))
    ax3.set_xlim(0.,600.)
    ax3.set_ylim(2000.,7000.)
    ax3.grid()


    ax3.set_xlabel('time (s)',fontsize=10,family='serif')

    ax1.set_ylabel('root bending\nmoment (kN-m)',fontsize=10,family='serif')
    ax2.set_ylabel('root bending\nmoment (kN-m)',fontsize=10,family='serif')
    ax3.set_ylabel('root bending\nmoment (kN-m)',fontsize=10,family='serif')

    ax1.set_title('freestream',fontsize=10,family='serif')
    ax2.set_title('fully waked, 4D downstream',fontsize=10,family='serif')
    ax3.set_title('fully waked, 10D downstream',fontsize=10,family='serif')

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')

    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')

    plt.subplots_adjust(top = 0.95, bottom = 0.1, right = 0.97, left = 0.15,
                hspace = 0.4, wspace = 0.5)


    plt.savefig('figures/loads_history.pdf',transparent=True)
    plt.show()
