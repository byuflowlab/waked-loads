
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


if __name__ == '__main__':
            print plt.rcParams['axes.prop_cycle'].by_key()['color']

            fig = plt.figure(1,figsize=[6.,2.])
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)

            bins_AEP = np.linspace(0.97,1.0,20)
            bins_dam = np.linspace(1.1,1.8,20)

            # axins = zoomed_inset_axes(ax1, 2.5, loc=4) # zoom-factor: 2.5, location: upper-left
            #
            # axins.get_xaxis().set_visible(False)
            # axins.get_yaxis().set_visible(False)
            # maxAEP = 46.319760162591045
            maxAEP = 45.2594551322
            # bins_AEP = np.linspace(0.9,1.0,20)


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



            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons1.2'
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

            ax1.hist(AEPfree,label='no damage\nconstraint',bins=bins_AEP,align='mid',ec='C0',alpha=0.5)
            ax2.hist(maxDfree,bins=bins_dam,align='mid',ec='C0',alpha=0.5)
            y = np.array([0.,100.])
            ax1.plot(np.array([np.mean(AEPfree),np.mean(AEPfree)]),y,color='C0',alpha=0.5)
            ax2.plot(np.array([np.mean(maxDfree),np.mean(maxDfree)]),y,color='C0',alpha=0.5)


            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_AEP2step1.15'

            fileAEP = folder+'/AEP_step2.txt'
            fileDAMAGE = folder+'/damage_step2.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP115 = AEP[0:100]
            AEP115 = AEP115/maxAEP
            maxD115 = np.zeros_like(AEP115)
            for i in range(len(AEP115)):
                        maxD115[i] = np.max(damage[i])
            ax1.hist(AEP115,label=r'$D_{\mathrm{max}} = 1.15$',bins=bins_AEP,align='mid',ec='C1',alpha=0.5)
            ax2.hist(maxD115,bins=bins_dam,align='mid',ec='C1',alpha=0.5)

            ax1.plot(np.array([np.mean(AEP115),np.mean(AEP115)]),y,color='C1',alpha=0.5)
            ax2.plot(np.array([np.mean(maxD115),np.mean(maxD115)]),y,color='C1',alpha=0.5)


            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons1.4'
            # fileAEP = folder+'/AEP.txt'
            # fileDAMAGE = folder+'/damage.txt'
            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_AEP2step1.5'

            fileAEP = folder+'/AEP_step2.txt'
            fileDAMAGE = folder+'/damage_step2.txt'
            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP15 = AEP[0:100]
            AEP15 = AEP15/maxAEP
            maxD15 = np.zeros_like(AEP15)
            for i in range(len(AEP15)):
                        maxD15[i] = np.max(damage[i])

            # ax1.hist(AEP,label=r'$D_{\mathrm{max}} = 1.5$',bins=bins_AEP,align='mid',ec='C2',fill=False,lw=3)
            # ax2.hist(maxD,bins=bins_dam,align='mid',ec='C2',fill=False,lw=3)




            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_aep'
            # fileAEP = folder+'/AEP.txt'
            # fileDAMAGE = folder+'/damage.txt'
            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_AEP2step1.3'
            fileAEP = folder+'/AEP_step2.txt'
            fileDAMAGE = folder+'/damage_step2.txt'
            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)
            AEP13 = AEP[0:100]
            AEP13 = AEP13/maxAEP
            maxD13 = np.zeros_like(AEP13)
            for i in range(len(AEP13)):
                        maxD13[i] = np.max(damage[i])
            lav = np.array([77.,0.,75.])/256.
            # ax1.hist(AEP,label=r'$D_{\mathrm{max}} = 1.3$',bins=bins_AEP,align='mid',ec='C3',fill=False,lw=2)
            # ax2.hist(maxD,bins=bins_dam,align='mid',ec='C3',fill=False,lw=2)




            # bins_AEP = np.linspace(0.97,1.0,15)
            # bins_dam = np.linspace(1.1,1.8,15)
            #
            # ax1.hist([AEPfree,AEP15,AEP13,AEP115],bins=bins_AEP,color=['C0','C2','C3','C1'])
            # ax2.hist([maxDfree,maxD15,maxD13,maxD115],bins=bins_dam,color=['C0','C2','C3','C1'])





            ax1.set_ylabel('# optimal farms',family='serif',fontsize=10)
            ax1.set_xlabel('AEP',family='serif',fontsize=10)

            ax2.set_xlabel('maximum turbine damage',family='serif',fontsize=10)
            #
            # ax1.set_xticks((0.9,0.95,1.0))
            # ax1.set_xticklabels(('0.9','0.95','1'))

            ax1.set_xticks((0.97,0.98,0.99,1.0))
            ax1.set_xticklabels(('0.97','0.98','0.99','1'))

            ax1.set_yticks((0,50,100))
            ax1.set_yticklabels(('0','50','100'))

            ax2.set_xticks((1.2,1.5,1.8))
            ax2.set_xticklabels(('1.2','1.5','1.8'))

            ax2.set_yticks((0.,50,100))
            # ax2.set_yticks((0,10,20))
            #
            ax2.set_yticklabels(('','',''))

            ax1.set_ylim(0,100)
            ax2.set_ylim(0,100)
            ax1.set_xlim(0.97,1.0)
            ax2.set_xlim(1.1,1.8)


            bins_AEP = np.linspace(0.97,1.0,20)
            bins_dam = np.linspace(1.15,1.8,20)

            ax1.legend(loc=2,prop={'size': 8, 'family': 'serif'})
            plt.subplots_adjust(top = 0.97, bottom = 0.2, right = 0.98, left = 0.1,
            hspace = 0, wspace = 0.1)

            ax1.text(0.991,50,'mean',horizontalalignment='center',verticalalignment='center',color='C1',fontsize=8,family='serif')
            ax1.text(0.9845,50,'mean',horizontalalignment='center',verticalalignment='center',color='C0',fontsize=8,family='serif')

            # plt.savefig('make_plots/figures/opt_results_hist.pdf',transparent=True)
            plt.show()
