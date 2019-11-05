
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


if __name__ == '__main__':
            fig = plt.figure(1,figsize=[6.,3.])
            ax1 = plt.subplot(111)

            import matplotlib as mpl
            label_size = 8
            mpl.rcParams['xtick.labelsize'] = label_size
            mpl.rcParams['ytick.labelsize'] = label_size
            #
            fontProperties = {'family':'serif','size':8}
            ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
            ax1.set_yticklabels(ax1.get_yticks(), fontProperties)

            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_aep'
            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/10turbs_2dirs_AEP2step1.0'

            fileAEP = folder+'/AEP_step1.txt'
            fileDAMAGE = folder+'/damage_step1.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)

            AEP = AEP[0:100]
            maxAEP = 45.2594551322
            AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])

            ax1.scatter(maxD,AEP,color='C0',s=4,label='no damage constraint',edgecolors=None)
            # axins.plot(maxD,AEP,'o',color='C0',s=4)


            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons1.4'
            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/10turbs_2dirs_AEP2step1.0'

            fileAEP = folder+'/AEP_step2.txt'
            fileDAMAGE = folder+'/damage_step2.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP = AEP[0:100]
            AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])
            ax1.scatter(maxD,AEP,color='C1',s=4,label=r'$D_{\mathrm{max}} = 1.0$',edgecolors=None)


            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/10turbs_2dirs_AEP2step0.98'

            fileAEP = folder+'/AEP_step2.txt'
            fileDAMAGE = folder+'/damage_step2.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP = AEP[0:100]
            AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])
            ax1.scatter(maxD,AEP,color='C2',s=4,label=r'$D_{\mathrm{max}} = 0.98$',edgecolors=None)


            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/10turbs_2dirs_AEP2step0.96'

            fileAEP = folder+'/AEP_step2.txt'
            fileDAMAGE = folder+'/damage_step2.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP = AEP[0:100]
            AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])
            ax1.scatter(maxD,AEP,color='C3',s=4,label=r'$D_{\mathrm{max}} = 0.96$',edgecolors=None)


            ax1.set_ylabel('normalized AEP',family='serif',fontsize=10)
            ax1.set_xlabel('max damage',family='serif',fontsize=10)

            ax1.set_yticks((0.9,0.95,1.0))
            ax1.set_yticklabels(('0.9','0.95','1'))

            ax1.set_xticks((1.0,1.2,1.4))
            ax1.set_xticklabels(('1.0','1.2','1.4'))
            ax1.set_xlim(0.95,1.405)

            ax1.legend(loc=4,prop={'size': 8, 'family': 'serif'}, markerscale=3)
            plt.subplots_adjust(top = 0.98, bottom = 0.2, right = 0.98, left = 0.1,
            hspace = 0, wspace = 0.)
            plt.savefig('make_plots/figures/opt_resultsREVISED.pdf',transparent=True)
            plt.show()
