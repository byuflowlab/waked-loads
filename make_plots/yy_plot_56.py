
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':


            offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])

            #FAST
            FAST4 = np.array([1.6243462 , 2.23706532, 2.45623725, 1.72047904, 0.97329602,
                        0.71895858, 0.56868445, 0.71493929, 1.13677232])
            FAST7 = np.array([1.57561877, 1.83788699, 1.92442706, 1.61282128, 1.17602627,
                        0.83198067, 0.74419571, 0.84849158, 1.0692838 ])
            FAST10 = np.array([1.50852404, 1.62225657, 1.62647763, 1.49749714, 1.26562855,
                        0.98813759, 0.90353996, 0.90228533, 1.04610205])

            super4 = np.array([1.4573791 , 1.88582452, 1.97091453, 1.43147071, 0.77288803,
                        0.44653579, 0.37573997, 0.49523845, 0.7992288 ])
            super7 = np.array([1.42154452, 1.62544464, 1.61813723, 1.28689932, 0.86407124,
                        0.60127591, 0.53184493, 0.62117129, 0.82027955])
            super10 = np.array([1.37628117, 1.46026477, 1.4051825 , 1.18618008, 0.91973512,
                        0.73472741, 0.67266584, 0.72141541, 0.84804457])

            extra_offset = np.array([-3.0,-2.75,-2.5,-2.25,-2.,-1.75,-1.5,-1.25,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.])

            extra_super4 = np.array([1.15199201, 1.15199201, 1.15199204, 1.15199351, 1.15203804,
                        1.15285361, 1.16189682, 1.22226935, 1.05462605, 1.13756772,
                        1.15073016, 1.15192463, 1.1519898 , 1.15199196, 1.15199201,
                        1.15199201])
            extra_super7 = np.array([1.15199201, 1.15199222, 1.15199636, 1.15205524, 1.15264974,
                        1.15689378, 1.17817454, 1.25221614, 1.01358923, 1.11383204,
                        1.14475829, 1.15102188, 1.15189896, 1.15198561, 1.15199169,
                        1.15199199])
            extra_super10 = np.array([1.15199279, 1.15200095, 1.15207174, 1.15254979, 1.15505118,
                        1.16513051, 1.19607165, 1.26660066, 0.98935354, 1.08703603,
                        1.13236822, 1.14742326, 1.15116224, 1.15187385, 1.1519788 ,
                        1.15199085])

            FASTfree = 1.3571995070891925

            fig = plt.figure(1,figsize=[6.,2.5])
            ax1 = plt.subplot(131)
            ax2 = plt.subplot(132)
            ax3 = plt.subplot(133)


            import matplotlib as mpl
            label_size = 8
            mpl.rcParams['xtick.labelsize'] = label_size
            mpl.rcParams['ytick.labelsize'] = label_size

            fontProperties = {'family':'serif','size':8}
            ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
            ax1.set_yticklabels(ax1.get_yticks(), fontProperties)

            ax2.set_xticklabels(ax2.get_xticks(), fontProperties)
            ax2.set_yticklabels(ax2.get_yticks(), fontProperties)

            ax3.set_xticklabels(ax3.get_xticks(), fontProperties)
            ax3.set_yticklabels(ax3.get_yticks(), fontProperties)

            offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])

            ax1.plot(offset,FAST4,'o',label='SOWFA+FAST',color='C1',markersize=4)
            ax1.plot(offset,super4,'o',label='superimposed loads',color='C0',markersize=4)
            ax1.plot(np.array([-3.,-1.]),np.array([1.,1.])*FASTfree,'--',color='C1',label='freestream loads')
            ax1.plot(np.array([1.,3.]),np.array([1.,1.])*FASTfree,'--',color='C1')
            # ax3.legend(loc=2,prop={'family':'serif', 'size':8})

            ax1.set_title('7D',family='serif',fontsize=10)
            ax1.set_ylabel('damage',family='serif',fontsize=10)

            ax1.set_xlabel('offset (D)',family='serif',fontsize=10)


            ax2.plot(offset,FAST7,'or',color='C1',markersize=4)
            ax2.plot(offset,super7,'ob',color='C0',markersize=4)
            ax2.plot(np.array([-3.,-1.]),np.array([1.,1.])*FASTfree,'--',color='C1')
            ax2.plot(np.array([1.,3.]),np.array([1.,1.])*FASTfree,'--',color='C1')
            ax2.set_title('7D',family='serif',fontsize=10)
            ax2.set_xlabel('offset (D)',family='serif',fontsize=10)
            #

            ax3.plot(offset,FAST10,'or',color='C1',markersize=4,label='SOWFA+FAST')
            ax3.plot(offset,super10,'ob',color='C0',markersize=4,label='CCBlade+gravity')
            ax3.set_title('10D',family='serif',fontsize=10)
            ax3.plot(np.array([-3.,-1.]),np.array([1.,1.])*FASTfree,'--',color='C1',label='freestream\nSOWFA+FAST')
            ax3.plot(np.array([1.,3.]),np.array([1.,1.])*FASTfree,'--',color='C1')
            ax3.set_xlabel('offset (D)',family='serif',fontsize=10)

            ax3.legend(loc=2,prop={'family':'serif', 'size':8})


            ax1.set_yticks((0.5,1.0,1.5,2.0,2.5))
            ax1.set_yticklabels(('0.5','1','1.5','2','2.5'))
            ax2.set_yticks((0.5,1.0,1.5,2.0,2.5))
            ax2.set_yticklabels(('0.5','1','1.5','2','2.5'))
            ax3.set_yticks((0.5,1.0,1.5,2.0,2.5))
            ax3.set_yticklabels(('0.5','1','1.5','2','2.5'))


            ax1.set_xlim(-3.1,3.1)
            ax2.set_xlim(-3.1,3.1)
            ax3.set_xlim(-3.1,3.1)

            ax1.set_xticks((-3.,-2.,-1.,0.,1.,2.,3.))
            ax1.set_xticklabels(('-3','-2','-1','0','1','2','3'))
            ax2.set_xticks((-3.,-2.,-1.,0.,1.,2.,3.))
            ax2.set_xticklabels(('-3','-2','-1','0','1','2','3'))
            ax3.set_xticks((-3.,-2.,-1.,0.,1.,2.,3.))
            ax3.set_xticklabels(('-3','-2','-1','0','1','2','3'))
            # ax1.set_xticks((-1.,-0.5,0.,0.5,1.))
            # ax1.set_xticklabels(('-1','-0.5','0','0.5','1'))
            # ax2.set_xticks((-1.,-0.5,0.,0.5,1.))
            # ax2.set_xticklabels(('-1','-0.5','0','0.5','1'))
            # ax3.set_xticks((-1.,-0.5,0.,0.5,1.))
            # ax3.set_xticklabels(('-1','-0.5','0','0.5','1'))

            ax1.set_title('4D downstream',family='serif',fontsize=10)
            ax2.set_title('7D downstream',family='serif',fontsize=10)
            ax3.set_title('10D downstream',family='serif',fontsize=10)

            plt.subplots_adjust(top = 0.8, bottom = 0.2, right = 0.98, left = 0.1,
            hspace = 0, wspace = 0.2)

            ax1.set_ylim(0.3,3.)
            ax2.set_ylim(0.3,3.)
            ax3.set_ylim(0.3,3.)

            ax1.plot(extra_offset,extra_super4,'o',color='C0',markersize=4)
            ax2.plot(extra_offset,extra_super7,'o',color='C0',markersize=4)
            ax3.plot(extra_offset,extra_super10,'o',color='C0',markersize=4)

            plt.suptitle('5.6% TI',family='serif',fontsize=10)
            plt.savefig('make_plots/figures/fatigue_damage56.pdf',transparent=True)
            plt.show()
