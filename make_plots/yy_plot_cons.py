
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


if __name__ == '__main__':
            fig = plt.figure(1,figsize=[6.,3.])
            ax1 = plt.subplot(111)
            # axins = zoomed_inset_axes(ax1, 2.5, loc=4) # zoom-factor: 2.5, location: upper-left
            #
            # axins.get_xaxis().set_visible(False)
            # axins.get_yaxis().set_visible(False)

            import matplotlib as mpl
            label_size = 8
            mpl.rcParams['xtick.labelsize'] = label_size
            mpl.rcParams['ytick.labelsize'] = label_size
            #
            fontProperties = {'family':'serif','size':8}
            ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
            ax1.set_yticklabels(ax1.get_yticks(), fontProperties)

            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_aep'
            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_AEP2step1.3'

            fileAEP = folder+'/AEP_step1.txt'
            fileDAMAGE = folder+'/damage_step1.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)

            AEP = AEP[0:100]
            maxAEP = 46.319760162591045
            AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])

            ax1.scatter(maxD,AEP,color='C0',s=4,label='no damage constraint',edgecolors=None)
            # axins.plot(maxD,AEP,'o',color='C0',s=4)


            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons1.4'
            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_AEP2step1.5'

            fileAEP = folder+'/AEP_step2.txt'
            fileDAMAGE = folder+'/damage_step2.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP = AEP[0:100]
            AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])
            ax1.scatter(maxD,AEP,color='C2',s=4,label=r'$D_{\mathrm{max}} = 1.5$',edgecolors=None)


            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons1.4'
            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_AEP2step1.3'

            fileAEP = folder+'/AEP_step2.txt'
            fileDAMAGE = folder+'/damage_step2.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP = AEP[0:100]
            AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])
            lav = np.array([77.,0.,75.])/256.
            ax1.scatter(maxD,AEP,color='C3',s=4,label=r'$D_{\mathrm{max}} = 1.3$',edgecolors=None)
            # axins.plot(maxD,AEP,'o',color='C1',s=4)


            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons1.2'
            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_AEP2step1.15'

            fileAEP = folder+'/AEP_step2.txt'
            fileDAMAGE = folder+'/damage_step2.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP = AEP[0:100]
            AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])
            ax1.scatter(maxD,AEP,color='C1',s=4,label=r'$D_{\mathrm{max}} = 1.15$',edgecolors=None)
            # axins.plot(maxD,AEP,'o',color='C3',s=4)


            ax1.set_ylabel('normalized AEP',family='serif',fontsize=10)
            ax1.set_xlabel('max damage',family='serif',fontsize=10)

            # ax1.set_yticks((0.6,0.7,0.8,0.9,1.0))
            # ax1.set_yticklabels(('0.6','0.7','0.8','0.9','1'))
            ax1.set_yticks((0.9,0.95,1.0))
            ax1.set_yticklabels(('0.9','0.95','1'))

            ax1.set_xticks((1.2,1.4,1.6,1.8))
            ax1.set_xticklabels(('1.2','1.4','1.6','1.8'))

            # axins.set_xlim(1.1,1.25)
            # axins.set_ylim(0.98,1.0)
            # from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            # mark_inset(ax1, axins, loc1=1, loc2=2, fc="none", ec="0.5")
            ax1.legend(loc=4,prop={'size': 8, 'family': 'serif'}, markerscale=3)
            plt.subplots_adjust(top = 0.98, bottom = 0.2, right = 0.98, left = 0.1,
            hspace = 0, wspace = 0.)
            # plt.savefig('make_plots/figures/opt_results4.pdf',transparent=True)
            plt.show()
