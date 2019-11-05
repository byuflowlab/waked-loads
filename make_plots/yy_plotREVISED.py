
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

            """TI = 0.11"""
            # #FAST
            # FAST4 = np.array([1.35137908, 1.71541373, 1.98567876, 1.3984026 , 0.94911793,
            #             0.67718382, 0.64275145, 0.77966465, 1.02635052])
            # FAST7 = np.array([1.32492845, 1.52989691, 1.62069779, 1.34037397, 1.03365472,
            #             0.84344457, 0.78222138, 0.83065681, 1.00642498])
            # FAST10 = np.array([1.29584224, 1.39422187, 1.411617  , 1.22741363, 0.95970128,
            #             0.90070624, 0.87901633, 0.87618861, 1.01573045])
            #
            # FASTfree = 1.0875922539496865
            #
            # CC4 = np.array([ 0.95747018,  0.95747027,  0.95747221,  0.95750427,  0.9578723 ,
            #             0.96080659,  0.97694552,  1.03747398,  1.18489344,  1.37103782,
            #             1.38923034,  1.10051281,  0.71421996,  0.47887851,  0.41644526,
            #             0.49104844,  0.67557731,  0.84624982,  0.92904837,  0.95256209,
            #             0.9568804 ,  0.95742037,  0.95746722,  0.95747006,  0.95747018])
            # CC7 = np.array([ 0.95747551,  0.95750993,  0.95771282,  0.95867984,  0.96238909,
            #             0.97374232,  1.00102357,  1.05053285,  1.11074008,  1.13747552,
            #             1.08080282,  0.94873053,  0.78677212,  0.66394124,  0.611998  ,
            #             0.63875774,  0.71989056,  0.81568851,  0.89061773,  0.93245934,
            #             0.94995787,  0.95563939,  0.95710602,  0.95741093,  0.95746229])
            # CC10 = np.array([ 0.95763661,  0.95807379,  0.95938526,  0.96277369,  0.97024496,
            #             0.98406951,  1.00479933,  1.02786568,  1.04114537,  1.02841865,
            #             0.98072148,  0.9062183 ,  0.82707527,  0.7660368 ,  0.73680109,
            #             0.74232147,  0.77623955,  0.82537443,  0.87442203,  0.91247844,
            #             0.93640694,  0.94890693,  0.95443262,  0.95652688,  0.95721321])

            """TI = 0.056"""
            #FAST
            FAST4 = np.array([1.36776643, 1.89714315, 2.09654972, 1.47612022, 0.83967825,
                        0.61750543, 0.48535682, 0.60648216, 0.95753692])
            FAST7 = np.array([1.32784831, 1.55496394, 1.63731782, 1.37968735, 1.00741318,
                        0.71084738, 0.63311522, 0.71779105, 0.90093821])
            FAST10 = np.array([1.27216204, 1.37212241, 1.38079224, 1.27542983, 1.079142  ,
                        0.8413648 , 0.76696968, 0.76236284, 0.88142521])

            FASTfree = 1.140557544276966

            CC4 = np.array([ 0.95747018,  0.95747018,  0.95747021,  0.95747147,  0.95750966,
                        0.95820905,  0.96596451,  1.01777584,  1.22129889,  1.59890394,
                        1.68786962,  1.24089428,  0.67875156,  0.39057955,  0.32215792,
                        0.41805318,  0.66867037,  0.87807253,  0.94570862,  0.95644108,
                        0.95741522,  0.95746838,  0.95747015,  0.95747018,  0.95747018])
            CC7 = np.array([ 0.95747019,  0.95747037,  0.95747392,  0.95752442,  0.95803436,
                        0.96167573,  0.9799473 ,  1.04369043,  1.19090556,  1.37254094,
                        1.37708451,  1.10316531,  0.7439813 ,  0.51576111,  0.4516839 ,
                        0.52299544,  0.6862808 ,  0.84457932,  0.92636094,  0.95157261,
                        0.95667916,  0.9573943 ,  0.95746497,  0.95746993,  0.95747017])
            CC10 = np.array([ 0.95747086,  0.95747785,  0.95753859,  0.95794884,  0.96009639,
                        0.96875746,  0.99539508,  1.05643818,  1.15268574,  1.23000079,
                        1.19044521,  1.00936853,  0.78394253,  0.62480823,  0.56896169,
                        0.60658388,  0.70956206,  0.82490112,  0.90454021,  0.94147693,
                        0.95374604,  0.95679372,  0.95737385,  0.95745941,  0.95746924])



            offsetFAST = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
            offsetCC = np.linspace(-3.,3.,25)

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

            ax1.plot(offsetFAST,FAST4,'o',label='SOWFA+FAST',color='C1',markersize=4)
            ax1.plot(offsetCC,CC4,'o',label='superimposed loads',color='C0',markersize=4)
            ax1.plot(np.array([-3.,-1.]),np.array([1.,1.])*FASTfree,'--',color='C1',label='freestream loads')
            ax1.plot(np.array([1.,3.]),np.array([1.,1.])*FASTfree,'--',color='C1')
            # ax3.legend(loc=2,prop={'family':'serif', 'size':8})

            ax1.set_title('7D',family='serif',fontsize=10)
            ax1.set_ylabel('damage',family='serif',fontsize=10)

            ax1.set_xlabel('offset (D)',family='serif',fontsize=10)


            ax2.plot(offsetFAST,FAST7,'or',color='C1',markersize=4)
            ax2.plot(offsetCC,CC7,'ob',color='C0',markersize=4)
            ax2.plot(np.array([-3.,-1.]),np.array([1.,1.])*FASTfree,'--',color='C1')
            ax2.plot(np.array([1.,3.]),np.array([1.,1.])*FASTfree,'--',color='C1')
            ax2.set_title('7D',family='serif',fontsize=10)
            ax2.set_xlabel('offset (D)',family='serif',fontsize=10)
            #

            ax3.plot(offsetFAST,FAST10,'or',color='C1',markersize=4,label='SOWFA+FAST')
            ax3.plot(offsetCC,CC10,'ob',color='C0',markersize=4,label='CCBlade+gravity')
            ax3.set_title('10D',family='serif',fontsize=10)
            ax3.plot(np.array([-3.,-1.]),np.array([1.,1.])*FASTfree,'--',color='C1',label='freestream\nSOWFA+FAST')
            ax3.plot(np.array([1.,3.]),np.array([1.,1.])*FASTfree,'--',color='C1')
            ax3.set_xlabel('offset (D)',family='serif',fontsize=10)

            ax3.legend(loc=1,prop={'family':'serif', 'size':7})


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

            ax1.set_ylim(0.,2.6)
            ax2.set_ylim(0.,2.6)
            ax3.set_ylim(0.,2.6)


            plt.suptitle('5.6% TI',family='serif',fontsize=10)
            plt.savefig('make_plots/figures/fatigue_damage56REVISED.pdf',transparent=True)
            plt.show()
