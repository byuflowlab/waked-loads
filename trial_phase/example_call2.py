
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as Time
from calc_fatigue_exp import *
# from calc_fatigue_rotor import *


if __name__ == '__main__':

        # T11
        #paths to the FAST output files
        filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
        filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
        # T5.6
        # filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C464_W8_T5.6_P0.0_m2D_L0/Model.out'
        # filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C437_W8_T5.6_P0.0_4D_L0/Model.out'
        # filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C455_W8_T5.6_P0.0_10D_L0/Model.out'
        #
        run = True
        run = False
        if run == True:
                N=24001
                # # #
                TI = 0.11
                print 'setup'
                s = Time.time()
                atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed = setup_atm(filename_free,filename_close,filename_far,TI=TI,N=N)
                print Time.time()-s

                sep = np.array([4.,7.,10.])
                # sep = np.array([10.])
                offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])

                start_time = Time.time()
                for i in range(len(sep)):
                        damage = np.zeros_like(offset)
                        for j in range(len(offset)):
                                # print 'separation: ', sep[i]
                                # print 'offset: ', offset[j]
                                turbineX = np.array([0.,126.4*sep[i]])
                                turbineY = np.array([0.,126.4])*offset[j]
                                # windDirections = np.array([270.,270.])
                                # windFrequencies = np.array([0.25,0.75])
                                windDirections = np.array([270.])
                                windFrequencies = np.array([1.])
                                s = Time.time()
                                damage[j] = farm_damage(turbineX,turbineY,windDirections,windFrequencies,atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed,TI=TI,N=N)[1]
                                # print "iteration time: ", (Time.time()-s)/2.
                                print damage[j]
                        print 'separation: ', sep[i]
                        print 'damage: ', repr(damage)
                print "time to run: ", (Time.time()-start_time)
                #

        offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])

        FAST4 = np.array([0.7479882 , 0.79418754, 0.91482836, 0.95437402, 0.9302589 ,0.96215701, 0.95952926, 0.86900191, 0.77281552])
        FAST7 = np.array([0.77931478, 0.82502943, 0.86154222, 0.89375726, 0.94159877, 0.8970284 , 0.87188754, 0.86078396, 0.81570256])
        FAST10 = np.array([0.7664489 , 0.85394419, 0.82298162, 0.83205289, 0.84391592, 0.85639593, 0.86367084, 0.82085985, 0.78787999])

        FASTfree = np.ones_like(FAST4)*0.6373399523538535

        #2* rad
        # super4 = np.array([0.67600654, 0.81147088, 0.94197874, 1.00382115, 1.0227498 , 1.0020521 , 0.94354412, 0.80575476, 0.7022973 ])
        # super7 = np.array([0.81066926, 0.96789703, 0.96769521, 0.9762949 , 0.95764617, 0.97673322, 0.95440725, 0.91732906, 0.81578342])
        # super10 = np.array([0.91137728, 0.93595487, 0.94413618, 0.92833491, 0.89210531, 0.93120719, 0.92008385, 0.91197694, 0.90520249])

        #1.75*rad
       #  super4 = np.array([0.64407964, 0.77874452, 0.90250068, 1.00258973, 1.0227498 ,
       #                          1.00371455, 0.89010514, 0.77154326, 0.65945321])
       #  super7 = np.array([0.76890235, 0.90406207, 0.964592  , 0.9762949 , 0.95764617,
       #                          0.97673322, 0.95088826, 0.83786392, 0.78751689])
       #  super10 = np.array([0.8470996 , 0.93515408, 0.94413618, 0.92833491, 0.89210531,
       # 0.93120719, 0.92008385, 0.90821004, 0.83267762])


        #2* rad
        # super4 = np.array([0.67600654, 0.81147088, 0.94197874, 1.00382115, 1.0227498 , 1.0020521 , 0.94354412, 0.80575476, 0.7022973 ])
        # super7 = np.array([0.81066926, 0.96789703, 0.96769521, 0.9762949 , 0.95764617, 0.97673322, 0.95440725, 0.91732906, 0.81578342])
        # super10 = np.array([0.91137728, 0.93595487, 0.94413618, 0.92833491, 0.89210531, 0.93120719, 0.92008385, 0.91197694, 0.90520249])

        #1.75*rad
        # super4 = np.array([0.6465743 , 0.78860154, 0.90094437, 1.00180077, 1.0227498 , 1.00450329, 0.89131835, 0.77691189, 0.6600066 ])
        # super7 = np.array([0.77986687, 0.90528373, 0.96459293, 0.9762949 , 0.95764617, 0.97673322, 0.95207005, 0.83987861, 0.79218427])
        # super10 = np.array([0.84477878, 0.93358042, 0.94413618, 0.92833491, 0.89210531, 0.93120719, 0.92008385, 0.9078248 , 0.83510624])
        # super4 = np.array([0.64407964, 0.77874452, 0.90250068, 1.00258973, 1.0227498 ,
        #                        1.00371455, 0.89010514, 0.77154326, 0.65945321])
        # super7 = np.array([0.76890235, 0.90406207, 0.964592  , 0.9762949 , 0.95764617,
        #                        0.97673322, 0.95088826, 0.83786392, 0.78751689])
        # super10 = np.array([0.8470996 , 0.93515408, 0.94413618, 0.92833491, 0.89210531,
        #                         0.93120719, 0.92008385, 0.90821004, 0.83267762])


        #1.5*rad
        # super4 = np.array([0.63079717, 0.70790713, 0.82569682, 1.00033454, 1.0227498 , 0.97183418, 0.80606136, 0.7054441 , 0.64222151])
        # super7 = np.array([0.70187135, 0.81749768, 0.92617547, 0.9762949 , 0.95764617, 0.97673322, 0.93479795, 0.76797933, 0.71548929])
        # super10 = np.array([0.754343  , 0.90503389, 0.94413618, 0.92833491, 0.89210531, 0.93120719, 0.92008385, 0.85395571, 0.75010338])
        # super4 = np.array([0.67600654, 0.81147088, 0.94197874, 1.00382115, 1.0227498 , 1.0020521 , 0.94354412, 0.80575476, 0.7022973 ])
        # super7 = np.array([0.81066926, 0.96789703, 0.96769521, 0.9762949 , 0.95764617, 0.97673322, 0.95440725, 0.91732906, 0.81578342])
        # super10 = np.array([0.91137728, 0.93595487, 0.94413618, 0.92833491, 0.89210531, 0.93120719, 0.92008385, 0.91197694, 0.90520249])


        #1.75
        # super4 = np.array([0.64407964, 0.77874452, 0.90250068, 1.00258973, 1.0227498 ,
        #         1.00371455, 0.89010514, 0.77154326, 0.65945321])
        # super7 = np.array([0.76890235, 0.90406207, 0.964592  , 0.9762949 , 0.95764617,
        #                         0.97673322, 0.95088826, 0.83786392, 0.78751689])
        # super10 = np.array([0.8470996 , 0.93515408, 0.94413618, 0.92833491, 0.89210531,
        #         0.93120719, 0.92008385, 0.90821004, 0.83267762])

        #1.5
        # super4 = np.array([0.62599883, 0.69708798, 0.82447868, 1.00033494, 1.0227498 ,
        #         0.97222414, 0.80757025, 0.69811344, 0.63900781])
        # super7 = np.array([0.69645718, 0.82292118, 0.92378866, 0.9762949 , 0.95764617,
        #         0.97673322, 0.93400189, 0.7667308 , 0.71156269])
        # super10 = np.array([0.7494355 , 0.90185501, 0.94413618, 0.92833491, 0.89210531,
        #         0.93120719, 0.92008385, 0.85156449, 0.74954593])



        # super4 = np.array([0.66751594, 0.79021415, 0.90292716, 0.97551056, 1.02875262,
        #         0.96643025, 0.88286043, 0.78291856, 0.68567893])
        # super7 = np.array([0.79894965, 0.9482317 , 0.93182298, 0.9533205 , 0.95282391,
        #         0.9440693 , 0.92759176, 0.87469453, 0.79327854])
        # super10 = np.array([0.90978908, 0.93595487, 0.94413618, 0.92833491, 0.89210531,
        #         0.93120719, 0.92008385, 0.91197694, 0.90557057])

                #p1
       #  super4 = np.array([0.67097228, 0.80982836, 0.94157487, 1.00382115, 1.0227498 ,
       # 1.0020521 , 0.94353668, 0.80448997, 0.6952231 ])
       #  super7 = np.array([0.80666315, 0.96512686, 0.96769521, 0.9762949 , 0.95764617,
       # 0.97673322, 0.95440725, 0.91534404, 0.8132822 ])
       #  super10 = np.array([0.90978908, 0.93595487, 0.94413618, 0.92833491, 0.89210531,
       # 0.93120719, 0.92008385, 0.91197694, 0.90557057])

       #  super4 = np.array([0.68484922, 0.80607067, 0.93822447, 1.00487786, 1.02402016,
       #          1.00331829, 0.93255572, 0.79346788, 0.69048914])
       #  super7 = np.array([0.80371251, 0.96695315, 0.96797935, 0.97656631, 0.95749718,
       #          0.97699201, 0.95470226, 0.9066866 , 0.80572874])
       #  super10 = np.array([0.90978908, 0.93595487, 0.94413618, 0.92833491, 0.89210531,
       # 0.93120719, 0.92008385, 0.91197694, 0.90557057])

        # super4 = np.array([0.68484922, 0.80607067, 0.93822447, 1.00487786, 1.02402016,
        #         1.00331829, 0.93255572, 0.79346788, 0.69048914])
        # super7 = np.array([0.80755671, 0.96695007, 0.9678074 , 0.97640206, 0.957743  ,
        #         0.97683541, 0.95452372, 0.91210784, 0.80618743])
        # super10 = np.array([0.91263047, 0.93549962, 0.94368804, 0.92788719, 0.89166757,
        #         0.93076213, 0.91962489, 0.91149129, 0.90340388])



        #2
       #  super4 = np.array([0.69798212, 0.80002447, 0.93622003, 1.01851485, 1.02303877,
       #          1.04213658, 0.94348549, 0.81829593, 0.72672876])
       #  super7 = np.array([0.80329802, 0.91122831, 0.95506371, 0.97864714, 0.95789985,
       # 0.97810997, 0.93600121, 0.95274938, 0.80799581])
       #  super10 = np.array([0.92148375, 0.92088275, 0.91786632, 0.93059056, 0.89101762,
       #          0.92375509, 0.91782382, 0.93273836, 0.90204759])

        #1.75
        super4 = np.array([0.65930373, 0.76562347, 0.89607523, 1.01763087, 1.02303877,
                1.04326872, 0.8828747 , 0.77765655, 0.69103651])
        super7 = np.array([0.7707067 , 0.84334717, 0.95397031, 0.97864714, 0.95789985,
                0.97810997, 0.94074128, 0.89684823, 0.77367233])
        super10 = np.array([0.87810422, 0.92012285, 0.91786632, 0.93059056, 0.89101762,
                0.92375509, 0.91782382, 0.93339593, 0.83084818])



        fig = plt.figure(1,figsize=[6.5,2.5])
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

        fig2 = plt.figure(2,figsize=[6.5,2.5])
        ax12 = plt.subplot(131)
        ax22 = plt.subplot(132)
        ax32 = plt.subplot(133)

        import matplotlib as mpl
        label_size = 6
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
        ax1.plot(offset,FAST4,'o',label='SOWFA+FAST',color='C3')
        ax1.plot(offset,super4,'o',label='superimposed',color='C0')
        ax12.plot(offset,(-FAST4+super4)/FAST4*100.,'o')
        ax1.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black',label='freestream loads')
        # ax1.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0',label='freestream superimposed')
        ax1.legend(loc=3,prop={'family':'serif', 'size':8})

        ax1.set_title('7D',family='serif',fontsize=8)
        ax1.set_ylabel('damage',family='serif',fontsize=8)

        ax1.set_xlabel('offset (D)',family='serif',fontsize=8)

        ax2.plot(offset,FAST7,'or',color='C3')
        ax2.plot(offset,super7,'ob',color='C0')
        ax22.plot(offset,(-FAST7+super7)/FAST7*100.,'o')
        # ax2.plot(offset,FASTfree,'ok')
        # ax2.plot(offset,freestreamCC,'og')
        ax2.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
        # ax2.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0')
        ax2.set_title('7D',family='serif',fontsize=8)
        ax2.set_xlabel('offset (D)',family='serif',fontsize=8)
        #
        ax3.plot(offset,FAST10,'or',color='C3')
        ax3.plot(offset,super10,'ob',color='C0')
        ax32.plot(offset,(-FAST10+super10)/FAST10*100.,'o')
        # ax3.plot(offset,FASTfree,'ok')
        # ax3.plot(offset,freestreamCC,'og')
        ax3.set_title('10D',family='serif',fontsize=8)
        ax3.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
        # ax3.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0')


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

        plt.subplots_adjust(top = 0.99, bottom = 0.2, right = 0.98, left = 0.1,
                    hspace = 0, wspace = 0.1)

        # ax1.set_ylim(0.6,1.1)
        # ax2.set_ylim(0.6,1.1)
        # ax3.set_ylim(0.6,1.1)

        ax12.set_yticks((-16.,0.,16.))
        ax12.set_yticklabels(('-16','0','16'))
        ax22.set_yticks((-16.,0.,16.))
        ax22.set_yticklabels(('','',''))
        ax32.set_yticks((-16.,0.,16.))
        ax32.set_yticklabels(('','',''))

        # plt.suptitle('1.5')
        # plt.savefig('fatigue_damage7D.pdf',transparent=True)
        plt.show()
