
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_aep'

            fileAEP = folder+'/AEP.txt'
            fileDAMAGE = folder+'/damage.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)

            print damage[0][0]

            AEP = AEP[0:100]
            AEP = AEP/np.max(AEP)
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])

            fig = plt.figure(1,figsize=[6.,2.5])
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)

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

            ax1.plot(maxD,AEP,'o',color='C0',markersize=1)
            ax1.set_ylabel('normalized AEP',family='serif',fontsize=10)
            ax1.set_xlabel('max damage',family='serif',fontsize=10)

            ax1.set_yticks((0.6,0.7,0.8,0.9,1.0))
            ax1.set_yticklabels(('0.6','0.7','0.8','0.9','1'))

            ax1.set_xticks((1.,1.2,1.4,1.6,1.8,2.0))
            ax1.set_xticklabels(('1','1.2','1.4','1.6','1.8','2'))

            bins = np.linspace(0.97,1.0001,20)
            ax2.hist(AEP,bins=bins,color='C0')





            #


            #
            #
            # ax1.set_xlim(-3.1,3.1)

            #
            # ax1.set_xticks((-3.,-2.,-1.,0.,1.,2.,3.))
            # ax1.set_xticklabels(('-3','-2','-1','0','1','2','3'))

            #
            # ax1.set_title('4D downstream',family='serif',fontsize=10)

            #
            # plt.subplots_adjust(top = 0.8, bottom = 0.2, right = 0.98, left = 0.1,
            # hspace = 0, wspace = 0.2)
            #
            # ax1.set_ylim(0.5,2.5)
            #
            # plt.suptitle('11% TI',family='serif',fontsize=10)
            # # plt.savefig('make_plots/figures/fatigue_damage11.pdf',transparent=True)
            plt.show()
