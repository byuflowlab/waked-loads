
import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':
            AEP = np.array([])
            maxDamage = np.array([])

            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_aep'
            fileAEP = folder+'/AEP.txt'
            fileDAMAGE = folder+'/damage.txt'
            aep = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            ind = np.argmax(aep)
            AEP = np.append(AEP,aep)
            maxD = np.zeros(len(aep))
            for i in range(len(maxD)):
                        maxD[i] = np.max(damage[i])
            maxDamage = np.append(maxDamage,maxD)


            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_damage'
            fileAEP = folder+'/AEP.txt'
            fileDAMAGE = folder+'/damage.txt'
            aep = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            ind = np.argmax(aep)
            AEP = np.append(AEP,aep)
            maxD = np.zeros(len(aep))
            for i in range(len(maxD)):
                        maxD[i] = np.max(damage[i])
            maxDamage = np.append(maxDamage,maxD)

            cons = np.array([1.2,1.21,1.22,1.225,1.23,1.24,1.25,1.26,1.27,1.275,1.28,1.29,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7])
            for k in range(len(cons)):
                        folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons%s'%cons[k]
                        fileAEP = folder+'/AEP.txt'
                        fileDAMAGE = folder+'/damage.txt'
                        aep = np.loadtxt(fileAEP)
                        damage = np.loadtxt(fileDAMAGE)
                        ind = np.argmax(aep)
                        AEP = np.append(AEP,aep)
                        maxD = np.zeros(len(aep))
                        for i in range(len(maxD)):
                                    maxD[i] = np.max(damage[i])
                        maxDamage = np.append(maxDamage,maxD)

            AEP = AEP/np.max(AEP)

            inds = np.argsort(AEP)
            aep = np.zeros_like(AEP)
            dam = np.zeros_like(AEP)
            for k in range(len(inds)):
                        aep[k] = AEP[inds[k]]
                        dam[k] = maxDamage[inds[k]]
            aep = np.flip(aep)
            dam = np.flip(dam)
            print aep
            print dam


            print np.min(dam)

            def pareto(aep,dam):
                        p_aep = np.array(aep[0])
                        p_dam = np.array(dam[0])
                        run = True
                        loc = 0
                        while run == True:
                                    AEP = aep[loc+1:len(aep)]
                                    DAMAGE = dam[loc+1:len(aep)]
                                    d = np.argmax(DAMAGE)
                                    if DAMAGE[d] < np.any(p_dam):
                                                print 'here'
                                                p_dam = np.append(p_dam,DAMAGE[d])
                                                p_aep = np.append(p_aep,AEP[d])
                                    else:
                                                run = False
                        return p_aep, p_dam



            p_aep, p_dam = pareto(aep,dam)
            print 'aep_arr: ', p_aep
            print 'dam_arr: ', p_dam

            fig = plt.figure(1,figsize=[3.5,2.5])
            ax1 = plt.subplot(111)

            import matplotlib as mpl
            label_size = 8
            mpl.rcParams['xtick.labelsize'] = label_size
            mpl.rcParams['ytick.labelsize'] = label_size
            #
            fontProperties = {'family':'serif','size':8}
            ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
            ax1.set_yticklabels(ax1.get_yticks(), fontProperties)

            ax1.plot(dam,aep,'o',color='C0',markersize=1)
            ax1.plot(p_dam,p_aep,'o',color='C1',markersize=3)
            ax1.set_ylabel('normalized AEP',family='serif',fontsize=10)
            ax1.set_xlabel('max damage',family='serif',fontsize=10)

            ax1.set_yticks((0.6,0.7,0.8,0.9,1.0))
            ax1.set_yticklabels(('0.6','0.7','0.8','0.9','1'))

            ax1.set_xticks((1.,1.2,1.4,1.6,1.8,1.85))
            ax1.set_xticklabels(('1','1.2','1.4','1.6','1.8',''))

            # ax1.set_xlim(-3.1,3.1)

            #
            # ax1.set_xticks((-3.,-2.,-1.,0.,1.,2.,3.))
            # ax1.set_xticklabels(('-3','-2','-1','0','1','2','3'))

            #
            # ax1.set_title('4D downstream',family='serif',fontsize=10)

            #
            plt.subplots_adjust(top = 0.98, bottom = 0.2, right = 0.98, left = 0.15,
            hspace = 0, wspace = 0.2)
            #
            # ax1.set_ylim(0.5,2.5)
            #
            # # plt.savefig('make_plots/figures/fatigue_damage11.pdf',transparent=True)
            plt.show()
