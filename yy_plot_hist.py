
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


if __name__ == '__main__':


            fig = plt.figure(1,figsize=[6.,2.])
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)

            # axins = zoomed_inset_axes(ax1, 2.5, loc=4) # zoom-factor: 2.5, location: upper-left
            #
            # axins.get_xaxis().set_visible(False)
            # axins.get_yaxis().set_visible(False)
            maxAEP = 46.319760162591045
            bins_AEP = np.linspace(0.9,1.0,20)
            bins_dam = np.linspace(1.1,1.8,20)

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


            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons1.2'
            fileAEP = folder+'/AEP.txt'
            fileDAMAGE = folder+'/damage.txt'
            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP = AEP[0:100]
            # AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])

            ax1.hist(AEP,color='C3',label=r'$D_{\mathrm{max}} = 1.2$',bins=bins_AEP,alpha=0.5,align='right')
            ax2.hist(maxD,color='C3',bins=bins_dam,alpha=0.5,align='right')



            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons1.4'
            fileAEP = folder+'/AEP.txt'
            fileDAMAGE = folder+'/damage.txt'
            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP = AEP[0:100]
            # AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])

            ax1.hist(AEP,color='C1',label=r'$D_{\mathrm{max}} = 1.4$',bins=bins_AEP,alpha=0.5,align='right')
            ax2.hist(maxD,color='C1',bins=bins_dam,alpha=0.5,align='right')




            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_aep'

            fileAEP = folder+'/AEP.txt'
            fileDAMAGE = folder+'/damage.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)

            AEP = AEP[0:100]
            plt.hist(AEP)
            plt.show()
            # AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])


            ax1.hist(AEP,color='C0',label='no damage constraint',bins=bins_AEP,alpha=0.6,align='right')
            ax2.hist(maxD,color='C0',bins=bins_dam,alpha=0.6,align='right')








            ax1.set_ylabel('# optimal farms',family='serif',fontsize=10)
            ax1.set_xlabel('AEP',family='serif',fontsize=10)

            ax2.set_xlabel('maximum turbine damage',family='serif',fontsize=10)
            #
            ax1.set_xticks((0.9,0.95,1.0))
            ax1.set_xticklabels(('0.9','0.95','1'))

            ax1.set_yticks((0.,50,100))
            ax1.set_yticklabels(('0','50','100'))

            ax2.set_xticks((1.2,1.5,1.8))
            ax2.set_xticklabels(('1.2','1.5','1.8'))

            ax2.set_yticks((0.,50,100))
            ax2.set_yticklabels(('','',''))

            ax1.legend(loc=2,prop={'size': 8, 'family': 'serif'})
            plt.subplots_adjust(top = 0.98, bottom = 0.2, right = 0.98, left = 0.1,
            hspace = 0, wspace = 0.1)

            # plt.savefig('make_plots/figures/opt_results_hist.pdf',transparent=True)
            plt.show()
