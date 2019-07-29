import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl



if __name__=="__main__":

    fig = plt.figure(1,figsize=[3.,3.])
    label_size = 8
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    fontProperties = {'family':'serif','size':10}

    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs'
    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])
    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])
    AEP_n = AEP[0:200]
    maxD_n = maxD[0:200]
    ind = np.argmax(AEP_n)
    AEP_n = AEP_n[ind]
    maxD_n = maxD_n[ind]

    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs_cons76'
    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])
    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])
    AEP_76 = AEP[0:200]
    maxD_76 = maxD[0:200]
    ind = np.argmax(AEP_76)
    AEP_76 = AEP_76[ind]
    maxD_76 = maxD_76[ind]

    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs_cons78'
    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])
    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])
    AEP_78 = AEP[0:200]
    maxD_78 = maxD[0:200]
    ind = np.argmax(AEP_78)
    AEP_78 = AEP_78[ind]
    maxD_78 = maxD_78[ind]

    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs_cons80'
    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])
    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])
    AEP_80 = AEP[0:200]
    maxD_80 = maxD[0:200]
    ind = np.argmax(AEP_80)
    AEP_80 = AEP_80[ind]
    maxD_80 = maxD_80[ind]


    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs_cons81'
    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])
    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])
    AEP_81 = AEP[0:200]
    maxD_81 = maxD[0:200]
    ind = np.argmax(AEP_81)
    AEP_81 = AEP_81[ind]
    maxD_81 = maxD_81[ind]

    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs_cons82'
    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])
    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])
    AEP_82 = AEP[0:200]
    maxD_82 = maxD[0:200]
    ind = np.argmax(AEP_82)
    AEP_82 = AEP_82[ind]
    maxD_82 = maxD_82[ind]

    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs_cons83'
    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])
    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])
    AEP_83 = AEP[0:200]
    maxD_83 = maxD[0:200]
    ind = np.argmax(AEP_83)
    AEP_83 = AEP_83[ind]
    maxD_83 = maxD_83[ind]

    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs_cons84'
    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])
    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])
    AEP_84 = AEP[0:200]
    maxD_84 = maxD[0:200]
    ind = np.argmax(AEP_84)
    AEP_84 = AEP_84[ind]
    maxD_84 = maxD_84[ind]

    folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/NAWEA/10_2dirs_cons85'
    fileAEP = folder+'/AEP.txt'
    fileMAX = folder+'/maxDamage.txt'
    with open(fileAEP) as my_file:
        AEP = my_file.readlines()
    for i in range(len(AEP)):
        AEP[i] = float(AEP[i])
    with open(fileMAX) as my_file:
        maxD = my_file.readlines()
    for i in range(len(maxD)):
        maxD[i] = float(maxD[i])
    AEP_85 = AEP[0:200]
    maxD_85 = maxD[0:200]
    ind = np.argmax(AEP_85)
    AEP_85 = AEP_85[ind]
    maxD_85 = maxD_85[ind]

    ax1 = plt.subplot(111)
    ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
    ax1.set_yticklabels(ax1.get_yticks(), fontProperties)

    AEP_arr = np.array([AEP_76,AEP_78,AEP_80,AEP_81,AEP_82,AEP_83,AEP_84,AEP_85,AEP_n])
    maxD_arr = np.array([maxD_76,maxD_78,maxD_80,maxD_81,maxD_82,maxD_83,maxD_84,maxD_85,maxD_n])

    print maxD_arr

    ax1.plot(AEP_arr/max(AEP_arr),maxD_arr,'o',color='C0')
    ax1.plot(AEP_n/max(AEP_arr),maxD_n,'o',color='C1')
    ax1.set_xticks((0.98,0.99,1.0))
    ax1.set_xticklabels(('0.98','0.99','1.0'))
    ax1.set_yticks((0.75,0.785,0.82))
    ax1.set_yticklabels(('0.75','0.785','0.82'))

    ax1.set_xlabel('AEP',fontsize=10,family='serif')
    ax1.set_ylabel('max damage',fontsize=10,family='serif')

    plt.subplots_adjust(top = 0.97, bottom = 0.15, right = 0.97, left = 0.3,
                hspace = 0.5, wspace = 0.5)

    plt.savefig('figures/pareto.pdf',transparent=True)
    plt.show()
