
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':


            offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])

            #FAST
            FAST4 = np.array([1.59935589, 2.02135062, 2.32792286, 1.63152952, 1.10297124,
                        0.78831172, 0.75221594, 0.91869222, 1.21524211])
            FAST7 = np.array([1.5670413 , 1.80450217, 1.90586825, 1.56885522, 1.20783595,
                        0.98616423, 0.91930079, 0.98111826, 1.19195753])
            FAST10 = np.array([1.5328147 , 1.64428406, 1.66251464, 1.44099974, 1.12516766,
                        1.05627586, 1.03500727, 1.03485405, 1.2027435 ])


            #1.
            super4 = np.array([1.44772848, 1.64528894, 1.62408023, 1.25671714, 0.81392429,
                        0.55373936, 0.48654219, 0.58240332, 0.82094436])
            super7 = np.array([1.35386686, 1.36955335, 1.28276262, 1.11595543, 0.92211638,
                        0.78030686, 0.72464306, 0.76560594, 0.87466405])
            super10 = np.array([1.26921323, 1.24391976, 1.1767105 , 1.08062303, 0.98368636,
                        0.91261462, 0.88248936, 0.89584237, 0.94445881])

            #2.
            # super4 = np.array([1.52267836, 1.78579735, 1.82732895, 1.42828944, 0.83376908,
            #             0.49873008, 0.43065707, 0.52715041, 0.7677847 ])
            # super7 = np.array([1.41556673, 1.45592606, 1.37091342, 1.17242574, 0.92920841,
            #             0.75049444, 0.67856615, 0.71563338, 0.82976686])
            # super10 = np.array([1.31086388, 1.29097343, 1.21811164, 1.10536444, 0.98688606,
            #             0.89678549, 0.85420648, 0.86220525, 0.91169349])

            #3.
            # super4 = np.array([1.59133872, 1.9116802 , 2.04583143, 1.63861146, 0.8577424 ,
            #             0.4449419 , 0.38401263, 0.48694727, 0.72437663])
            # super7 = np.array([1.47756671, 1.54474828, 1.46519383, 1.23399001, 0.93674273,
            #             0.72062537, 0.63529787, 0.67044697, 0.78876674])
            # super10 = np.array([1.35331838, 1.33956837, 1.26131592, 1.13116126, 0.99015894,
            #             0.88089708, 0.82655337, 0.8299185 , 0.88042739])

            #4.
            # super4 = np.array([1.65443183, 2.01919265, 2.2805667 , 1.89479409, 0.88490817,
            #             0.39344698, 0.34477842, 0.45917911, 0.68842688])
            # super7 = np.array([1.53969606, 1.6357609 , 1.56584002, 1.30071301, 0.9446843 ,
            #             0.6909475 , 0.59476957, 0.62964032, 0.75133547])
            # super10 = np.array([1.39662838, 1.38974232, 1.30630978, 1.15803022, 0.99350236,
            #             0.86497566, 0.79956901, 0.79892994, 0.85053863])

            # super4 = np.
            # super7 = np.
            # super10 = np.

            extra_offset = np.array([-3.0,-2.75,-2.5,-2.25,-2.,-1.75,-1.5,-1.25,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.])
            extra_super4 = np.array([1.18643912, 1.18643929, 1.18644341, 1.18651213, 1.18730992,
                   1.19371254, 1.2287789 , 1.35743101, 0.96217584, 1.1248105 ,
                   1.17554658, 1.18512988, 1.18632939, 1.18643267, 1.18643885,
                   1.18643911])
            extra_super7 = np.array([1.18645068, 1.18652575, 1.18696915, 1.18908546, 1.19721254,
                   1.222123  , 1.28230335, 1.39384264, 0.91078713, 1.0492885 ,
                   1.13319212, 1.17010427, 1.18241404, 1.18563383, 1.18630775,
                   1.18642162])
            extra_super10 = np.array([1.18680378, 1.18776355, 1.19064998, 1.19814306, 1.21481159,
                   1.2461217 , 1.2946727 , 1.35287989, 0.93333617, 1.02186697,
                   1.094621  , 1.14244081, 1.16825788, 1.17991367, 1.18439477,
                   1.18587845])

            FASTfree = 1.288629410658714

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

            ax1.set_ylim(0.5,2.5)
            ax2.set_ylim(0.5,2.5)
            ax3.set_ylim(0.5,2.5)

            ax1.plot(extra_offset,extra_super4,'o',color='C0',markersize=4)
            ax2.plot(extra_offset,extra_super7,'o',color='C0',markersize=4)
            ax3.plot(extra_offset,extra_super10,'o',color='C0',markersize=4)

            plt.suptitle('11% TI',family='serif',fontsize=10)
            # plt.savefig('make_plots/figures/fatigue_damage11.pdf',transparent=True)
            plt.show()
