import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl



if __name__=="__main__":

    fig = plt.figure(1,figsize=[6.,6.])
    label_size = 8
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    fontProperties = {'family':'serif','size':10}

    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs'

    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    fileMEAN = folder+'/meanDamage.txt'

    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])

    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])

    AEP = AEP[0:200]
    maxD = maxD[0:200]

    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
    ax1.set_yticklabels(ax1.get_yticks(), fontProperties)
    ax2.set_xticklabels(ax2.get_xticks(), fontProperties)
    ax2.set_yticklabels(ax2.get_yticks(), fontProperties)
    ax3.set_xticklabels(ax1.get_xticks(), fontProperties)
    ax3.set_yticklabels(ax1.get_yticks(), fontProperties)
    ax4.set_xticklabels(ax2.get_xticks(), fontProperties)
    ax4.set_yticklabels(ax2.get_yticks(), fontProperties)


    m = np.max(AEP)

    bins = np.linspace(0.9,1.0001,20)
    ax1.hist(AEP/m,bins=bins,color='C0')
    ax1.set_xticks((0.9,0.95,1.0))
    ax1.set_xticklabels(('0.9','0.95','1.0'))
    ax1.set_xlabel('normalized AEP',fontsize=10,family='serif')
    ax1.set_yticks((0.,20.,40.))
    ax1.set_yticklabels(('','20','40'))

    ax2.plot(maxD,AEP/m,'o',markersize=2,color='C0')
    ax2.set_ylabel('normalized AEP',fontsize=10,family='serif')
    ax2.set_yticks((0.9,0.95,1.0))
    ax2.set_yticklabels(('0.9','0.95','1.0'))
    ax2.set_ylim(0.9,1.01)
    ax2.set_xticks((0.8,0.9,1.0))
    ax2.set_xticklabels(('0.8','0.9','1.0'))
    ax2.set_xlabel('maximum damage',fontsize=10,family='serif')



    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs_cons85'

    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    fileMEAN = folder+'/meanDamage.txt'

    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])

    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])

    AEP = AEP[0:200]
    maxD = maxD[0:200]

    ax3.hist(AEP/m,bins=bins,color='C1')
    ax3.set_xticks((0.9,0.95,1.0))
    ax3.set_xticklabels(('0.9','0.95','1.0'))
    ax3.set_xlabel('normalized AEP',fontsize=10,family='serif')
    ax3.set_yticks((0.,20.,40.))
    ax3.set_yticklabels(('','20','40'))

    ax4.plot(maxD,AEP/m,'o',markersize=2,color='C1')
    ax4.set_ylabel('normalized AEP',fontsize=10,family='serif')
    ax4.set_yticks((0.9,0.95,1.0))
    ax4.set_yticklabels(('0.9','0.95','1.0'))
    ax4.set_ylim(0.9,1.01)
    ax4.set_xticks((0.8,0.9,1.0))
    ax4.set_xticklabels(('0.8','0.9','1.0'))
    ax4.set_xlabel('maximum damage',fontsize=10,family='serif')

    ax2.set_xlim(0.75,1.0)
    ax4.set_xlim(0.75,1.0)







    plt.subplots_adjust(top = 0.97, bottom = 0.1, right = 0.97, left = 0.1,
                hspace = 0.5, wspace = 0.5)


    plt.savefig('figures/spread85.pdf',transparent=True)
    plt.show()
