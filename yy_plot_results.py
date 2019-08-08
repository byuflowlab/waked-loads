
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_aep'
            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_damage'
            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons1.27'

            fileAEP = folder+'/AEP.txt'
            fileDAMAGE = folder+'/damage.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)

            AEP = AEP[0:100]
            print max(AEP)
            maxAEP = 46.319760162591045
            AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])



            # print min(maxD)

            fig = plt.figure(1,figsize=[6.,2.5])
            ax1 = plt.subplot(131)
            ax2 = plt.subplot(132)
            ax3 = plt.subplot(133)

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
            ax3.set_xticklabels(ax1.get_xticks(), fontProperties)
            ax3.set_yticklabels(ax1.get_yticks(), fontProperties)

            ax1.plot(maxD,AEP,'o',color='C0',markersize=1)
            ax1.set_ylabel('normalized AEP',family='serif',fontsize=10)
            ax1.set_xlabel('max damage',family='serif',fontsize=10)


            binsAEP = np.linspace(0.6,1.0001,20)
            ax2.hist(AEP,bins=binsAEP,color='C0')

            binsDam = np.linspace(1.,2.,20)
            ax3.hist(maxD,bins=binsDam,color='C0')



            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_aep'
            folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_damage'
            # folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_cons1.27'

            fileAEP = folder+'/AEP.txt'
            fileDAMAGE = folder+'/damage.txt'

            AEP = np.loadtxt(fileAEP)
            damage = np.loadtxt(fileDAMAGE)

            AEP = AEP[0:100]
            print max(AEP)
            maxAEP = 46.319760162591045
            AEP = AEP/maxAEP
            maxD = np.zeros_like(AEP)
            for i in range(len(AEP)):
                        maxD[i] = np.max(damage[i])



            #
            fontProperties = {'family':'serif','size':8}
            ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
            ax1.set_yticklabels(ax1.get_yticks(), fontProperties)
            ax2.set_xticklabels(ax1.get_xticks(), fontProperties)
            ax2.set_yticklabels(ax1.get_yticks(), fontProperties)

            ax1.plot(maxD,AEP,'o',color='C1',markersize=1)
            ax1.set_ylabel('normalized AEP',family='serif',fontsize=10)
            ax1.set_xlabel('max damage',family='serif',fontsize=10)

            ax2.hist(AEP,bins=binsAEP,color='C1')
            ax3.hist(maxD,bins=binsDam,color='C1')

            # ax2.set_xticks((0.9,0.92,0.94,0.96,0.98,1.0))
            # ax2.set_xticklabels(('0.9','0.92','0.94','0.96','0.98','1'))




            ax1.set_yticks((0.6,0.7,0.8,0.9,1.0))
            ax1.set_yticklabels(('0.6','0.7','0.8','0.9','1'))
            # ax1.set_yticks((0.96,0.98,1.0))
            # ax1.set_yticklabels(('0.96','0.98','1'))

            ax1.set_xticks((1.,1.2,1.4,1.6,1.8,2.0))
            ax1.set_xticklabels(('1','1.2','1.4','1.6','1.8','2'))


            ax2.set_yticks((20,40,60,80,100))
            ax2.set_yticklabels(('20','40','60','80','100'))
            ax2.set_xticks((0.6,0.7,0.8,0.9,1.0))
            ax2.set_xticklabels(('0.6','0.7','0.8','0.9','1'))

            ax3.set_yticks((20,40,60,80,100))
            ax3.set_yticklabels(('20','40','60','80','100'))
            ax3.set_xticks((1.,1.25,1.5,1.75,2.))
            ax3.set_xticklabels(('1','1.25','1.5','1.75','2'))

            ax2.set_xlabel('normalized AEP',family='serif',fontsize=10)
            ax3.set_xlabel('max damage',family='serif',fontsize=10)

            plt.subplots_adjust(top = 0.98, bottom = 0.2, right = 0.98, left = 0.1,
            hspace = 0, wspace = 0.2)
            # # plt.savefig('make_plots/figures/fatigue_damage11.pdf',transparent=True)
            plt.show()
