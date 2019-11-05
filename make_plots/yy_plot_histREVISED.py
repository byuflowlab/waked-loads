
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


if __name__ == '__main__':
            # print plt.rcParams['axes.prop_cycle'].by_key()['color']

            fig = plt.figure(1,figsize=[6.,2.])
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)

            bins_AEP = np.linspace(0.94,1.0,20)
            bins_dam = np.linspace(0.94,1.4,20)

            print bins_dam

            maxAEP = 45.2594551322

            import matplotlib as mpl
            label_size = 8
            mpl.rcParams['xtick.labelsize'] = label_size
            mpl.rcParams['ytick.labelsize'] = label_size
            #
            fontProperties = {'family':'serif','size':8}
            ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
            ax1.set_yticklabels(ax1.get_yticks(), fontProperties)

            ax2.set_xticklabels(ax1.get_xticks(), fontProperties)
            ax2.set_yticklabels(ax1.get_yticks(), fontProperties)

            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/10turbs_2dirs_AEP2step1.0'
            fileAEP = folder+'/AEP_step1.txt'
            fileDAMAGE = folder+'/damage_step1.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEPfree = AEP[0:100]
            AEPfree = AEPfree/maxAEP

            maxDfree = np.zeros_like(AEPfree)
            for i in range(len(AEPfree)):
                        maxDfree[i] = np.max(damage[i])

            ax1.hist(AEPfree,label='no damage\nconstraint',bins=bins_AEP,align='left',ec='C0',alpha=0.5)
            ax2.hist(maxDfree,bins=bins_dam,align='left',ec='C0',alpha=0.5)
            y = np.array([0.,100.])
            ax1.plot(np.array([np.mean(AEPfree),np.mean(AEPfree)]),y,color='C0',alpha=0.5,linewidth=2)
            ax2.plot(np.array([np.mean(maxDfree),np.mean(maxDfree)]),y,color='C0',alpha=0.5,linewidth=2)




            fileAEP = folder+'/AEP_step2.txt'
            fileDAMAGE = folder+'/damage_step2.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEPcon = AEP[0:100]
            AEPcon = AEPcon/maxAEP



            maxDcon = np.zeros_like(AEPcon)
            for i in range(len(AEPcon)):
                        maxDcon[i] = np.max(damage[i])

            # print min(maxDfree)
            # print min(maxDcon)
            # print max(maxDfree)
            # print max(maxDcon)

            ax1.hist(AEPcon,label=r'$D_{\mathrm{max}} = 1.0$',bins=bins_AEP,align='left',ec='C1',alpha=0.5)
            ax2.hist(maxDcon,bins=bins_dam,align='left',ec='C1',alpha=0.5)

            ax1.plot(np.array([np.mean(AEPcon),np.mean(AEPcon)]),y,color='C1',alpha=0.5,linewidth=2)
            ax2.plot(np.array([np.mean(maxDcon),np.mean(maxDcon)]),y,color='C1',alpha=0.5,linewidth=2)


            ax1.set_ylabel('# optimal farms',family='serif',fontsize=10)
            ax1.set_xlabel('AEP',family='serif',fontsize=10)
            ax2.set_xlabel('maximum turbine damage',family='serif',fontsize=10)

            ax1.set_xticks((0.94,0.96,0.98,1.0))
            ax1.set_xticklabels(('0.94','0.96','0.98','1'))

            ax1.set_yticks((5,10,15,17.5,20))
            ax1.set_yticklabels(('5','10','15','...','100'))
            #
            ax2.set_xticks((1.0,1.2,1.4))
            ax2.set_xticklabels(('1.0','1.2','1.4'))
            #
            ax2.set_yticks((5,10,15,17.5,20))
            ax2.set_yticklabels(('','','','...',''))

            dbin_aep = bins_AEP[1]-bins_AEP[0]
            dbin_dam = bins_dam[1]-bins_dam[0]
            ax1.set_ylim(0,20)
            ax2.set_ylim(0,20)
            ax1.set_xlim(0.94-dbin_aep/2.,1.0-dbin_aep/2.)
            ax2.set_xlim(0.95-dbin_dam/2.,1.4-dbin_dam/2.)


            ax1.legend(loc=2,prop={'size': 8, 'family': 'serif'})
            plt.subplots_adjust(top = 0.97, bottom = 0.2, right = 0.98, left = 0.1,
            hspace = 0, wspace = 0.1)

            ax1.text(0.97275,14,'mean',horizontalalignment='center',verticalalignment='center',color='C1',fontsize=8,family='serif')
            ax1.text(0.986,14,'mean',horizontalalignment='center',verticalalignment='center',color='C0',fontsize=8,family='serif')

            plt.savefig('make_plots/figures/opt_results_histREVISED.pdf',transparent=True)
            plt.show()
