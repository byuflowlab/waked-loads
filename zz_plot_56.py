
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':


            offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])

            # FLAPWISE
            FAST4 = np.array([0.62301502, 0.76051433, 0.99362264, 1.06089069, 0.97718426,
                        0.96806468, 0.96699219, 0.77118321, 0.64583782])
            FAST7 = np.array([0.74998387, 0.83608668, 0.87085191, 0.91296148, 0.9531732 ,
                        0.97340949, 0.93045233, 0.85915521, 0.72402707])
            FAST10 = np.array([0.83042745, 0.85503197, 0.86062394, 0.88745736, 0.89488398,
                        0.89238626, 0.9146126 , 0.89057379, 0.7829162 ])

            # 2.0
            super4 = np.array([0.53482881, 0.56876244, 0.69613627, 0.91201709, 0.98309726,
                        0.92156986, 0.71182754, 0.5770518 , 0.55340153])
            super7 = np.array([0.58319188, 0.68502466, 0.7730332 , 0.87358835, 1.0020204 ,
                        0.94621898, 0.78526891, 0.65646819, 0.61263002])
            super10 = np.array([0.74956186, 0.83715753, 0.8796291 , 0.95643124, 0.89882598,
                        0.91136063, 0.90585764, 0.8566962 , 0.7561317 ])

            # 2.5
            # super4 = np.array([0.71064867, 0.71103565, 0.83170962, 0.91641062, 0.98309726,
            #             0.92650891, 0.80496412, 0.74548618, 0.7066731 ])
            # super7 = np.array([0.77446499, 0.81173529, 0.82194694, 0.87358835, 1.0020204 ,
            #             0.94621898, 0.84698879, 0.8113935 , 0.77186784])
            # super10 = np.array([0.88085427, 0.87520546, 0.8796291 , 0.95643124, 0.89882598,
            #             0.91136063, 0.90585764, 0.90830444, 0.88907596])

            # 2.6
            # super4 = np.array([0.73420732, 0.73214167, 0.84695489, 0.91641062, 0.98309726,
            #             0.92650891, 0.8222151 , 0.76524181, 0.74588359])
            # super7 = np.array([0.81572162, 0.82953803, 0.82194694, 0.87358835, 1.0020204 ,
            #             0.94621898, 0.84698879, 0.83411028, 0.81072535])
            # super10 = np.array([0.90498245, 0.87520546, 0.8796291 , 0.95643124, 0.89882598,
            #             0.91136063, 0.90585764, 0.90830444, 0.90849212])


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
            ax1.plot(offset,FAST4,'o',label='SOWFA+FAST',color='C1')
            ax1.plot(offset,super4,'o',label='superimposed',color='C0')
            # ax1.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black',label='freestream loads')
            ax1.legend(loc=3,prop={'family':'serif', 'size':8})

            ax1.set_title('7D',family='serif',fontsize=10)
            ax1.set_ylabel('damage',family='serif',fontsize=10)

            ax1.set_xlabel('offset (D)',family='serif',fontsize=10)

            ax2.plot(offset,FAST7,'or',color='C1')
            ax2.plot(offset,super7,'ob',color='C0')
            # ax2.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
            ax2.set_title('7D',family='serif',fontsize=10)
            ax2.set_xlabel('offset (D)',family='serif',fontsize=10)
            #
            ax3.plot(offset,FAST10,'or',color='C1')
            ax3.plot(offset,super10,'ob',color='C0')
            ax3.set_title('10D',family='serif',fontsize=10)
            # ax3.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
            ax3.set_xlabel('offset (D)',family='serif',fontsize=10)


            ax1.set_yticks((0.25,0.5,0.75,1.0,1.1))
            ax1.set_yticklabels(('0.25','0.5','0.75','1.0',''))
            ax2.set_yticks((0.25,0.5,0.75,1.0,1.1))
            ax2.set_yticklabels(('','','','',''))
            ax3.set_yticks((0.25,0.5,0.75,1.0,1.1))
            ax3.set_yticklabels(('','','','',''))


            ax1.set_xlim(-1.1,1.1)
            ax2.set_xlim(-1.1,1.1)
            ax3.set_xlim(-1.1,1.1)

            ax1.set_xticks((-1.,-0.5,0.,0.5,1.))
            ax1.set_xticklabels(('-1','-0.5','0','0.5','1'))
            ax2.set_xticks((-1.,-0.5,0.,0.5,1.))
            ax2.set_xticklabels(('-1','-0.5','0','0.5','1'))
            ax3.set_xticks((-1.,-0.5,0.,0.5,1.))
            ax3.set_xticklabels(('-1','-0.5','0','0.5','1'))

            ax1.set_title('4D downstream',family='serif',fontsize=10)
            ax2.set_title('7D downstream',family='serif',fontsize=10)
            ax3.set_title('10D downstream',family='serif',fontsize=10)

            plt.subplots_adjust(top = 0.8, bottom = 0.2, right = 0.98, left = 0.1,
            hspace = 0, wspace = 0.1)

            ax1.set_ylim(0.25,1.1)
            ax2.set_ylim(0.25,1.1)
            ax3.set_ylim(0.25,1.1)

            plt.suptitle('5.6% TI',family='serif',fontsize=10)
            # plt.savefig('make_plots/figures/fatigue_damage56.pdf',transparent=True)
            plt.show()
