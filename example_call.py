
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as Time
# from calc_fatigue_exp import *
from calc_fatigue_rotor import *


if __name__ == '__main__':

        # """full off the FAST data and make fatigue calculations"""
        # sep = np.array([4.,7.,10.])
        # offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
        # # sep = np.array([10.])
        # # offset = np.array([0.])
        # TI = 11.
        # freestream = False
        # for i in range(len(sep)):
        #         damage = np.zeros_like(offset)
        #         for j in range(len(offset)):
        #                 if freestream == True:
        #                         if TI == 11.:
        #                                 # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
        #                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C676_W8_T11.0_P0.0_m2D_L-1.0/Model.out'
        #                         if TI == 5.6:
        #                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C464_W8_T5.6_P0.0_m2D_L0/Model.out'
        #
        #                 else:
        #                         if TI == 11.:
        #                                 if sep[i] == 4.:
        #                                         if offset[j] == -1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C649_W8_T11.0_P0.0_4D_L-1.0/Model.out'
        #                                         if offset[j] == -0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C650_W8_T11.0_P0.0_4D_L-0.75/Model.out'
        #                                         if offset[j] == -0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C651_W8_T11.0_P0.0_4D_L-0.5/Model.out'
        #                                         if offset[j] == -0.25:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C652_W8_T11.0_P0.0_4D_L-0.25/Model.out'
        #                                         if offset[j] == 0.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        #                                         if offset[j] == 0.25:
        #                                                 filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C654_W8_T11.0_P0.0_4D_L0.25/Model.out'
        #                                         if offset[j] == 0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C655_W8_T11.0_P0.0_4D_L0.5/Model.out'
        #                                         if offset[j] == 0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C656_W8_T11.0_P0.0_4D_L0.75/Model.out'
        #                                         if offset[j] == 1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C657_W8_T11.0_P0.0_4D_L1.0/Model.out'
        #                                 if sep[i] == 7.:
        #                                         if offset[j] == -1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C658_W8_T11.0_P0.0_7D_L-1.0/Model.out'
        #                                         if offset[j] == -0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C659_W8_T11.0_P0.0_7D_L-0.75/Model.out'
        #                                         if offset[j] == -0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C660_W8_T11.0_P0.0_7D_L-0.5/Model.out'
        #                                         if offset[j] == -0.25:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C661_W8_T11.0_P0.0_7D_L-0.25/Model.out'
        #                                         if offset[j] == 0.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C662_W8_T11.0_P0.0_7D_L0/Model.out'
        #                                         if offset[j] == 0.25:
        #                                                 filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C663_W8_T11.0_P0.0_7D_L0.25/Model.out'
        #                                         if offset[j] == 0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
        #                                         if offset[j] == 0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C665_W8_T11.0_P0.0_7D_L0.75/Model.out'
        #                                         if offset[j] == 1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C666_W8_T11.0_P0.0_7D_L1.0/Model.out'
        #                                 if sep[i] == 10.:
        #                                         if offset[j] == -1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C667_W8_T11.0_P0.0_10D_L-1.0/Model.out'
        #                                         if offset[j] == -0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C668_W8_T11.0_P0.0_10D_L-0.75/Model.out'
        #                                         if offset[j] == -0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C669_W8_T11.0_P0.0_10D_L-0.5/Model.out'
        #                                         if offset[j] == -0.25:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C670_W8_T11.0_P0.0_10D_L-0.25/Model.out'
        #                                         if offset[j] == 0.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
        #                                         if offset[j] == 0.25:
        #                                                 filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C672_W8_T11.0_P0.0_10D_L0.25/Model.out'
        #                                         if offset[j] == 0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C673_W8_T11.0_P0.0_10D_L0.5/Model.out'
        #                                         if offset[j] == 0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C674_W8_T11.0_P0.0_10D_L0.75/Model.out'
        #                                         if offset[j] == 1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C675_W8_T11.0_P0.0_10D_L1.0/Model.out'
        #
        #                         if TI == 5.6:
        #                                 if sep[i] == 4.:
        #                                         if offset[j] == -1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C433_W8_T5.6_P0.0_4D_L-1.0/Model.out'
        #                                         if offset[j] == -0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C434_W8_T5.6_P0.0_4D_L-0.75/Model.out'
        #                                         if offset[j] == -0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C435_W8_T5.6_P0.0_4D_L-0.5/Model.out'
        #                                         if offset[j] == -0.25:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C436_W8_T5.6_P0.0_4D_L-0.25/Model.out'
        #                                         if offset[j] == 0.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C437_W8_T5.6_P0.0_4D_L0/Model.out'
        #                                         if offset[j] == 0.25:
        #                                                 filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C438_W8_T5.6_P0.0_4D_L0.25/Model.out'
        #                                         if offset[j] == 0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C439_W8_T5.6_P0.0_4D_L0.5/Model.out'
        #                                         if offset[j] == 0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C440_W8_T5.6_P0.0_4D_L0.75/Model.out'
        #                                         if offset[j] == 1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C441_W8_T5.6_P0.0_4D_L1.0/Model.out'
        #                                 if sep[i] == 7.:
        #                                         if offset[j] == -1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C442_W8_T5.6_P0.0_7D_L-1.0/Model.out'
        #                                         if offset[j] == -0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C443_W8_T5.6_P0.0_7D_L-0.75/Model.out'
        #                                         if offset[j] == -0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C444_W8_T5.6_P0.0_7D_L-0.5/Model.out'
        #                                         if offset[j] == -0.25:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C445_W8_T5.6_P0.0_7D_L-0.25/Model.out'
        #                                         if offset[j] == 0.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C446_W8_T5.6_P0.0_7D_L0/Model.out'
        #                                         if offset[j] == 0.25:
        #                                                 filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C447_W8_T5.6_P0.0_7D_L0.25/Model.out'
        #                                         if offset[j] == 0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C448_W8_T5.6_P0.0_7D_L0.5/Model.out'
        #                                         if offset[j] == 0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C449_W8_T5.6_P0.0_7D_L0.75/Model.out'
        #                                         if offset[j] == 1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C450_W8_T5.6_P0.0_7D_L1.0/Model.out'
        #                                 if sep[i] == 10.:
        #                                         if offset[j] == -1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C451_W8_T5.6_P0.0_10D_L-1.0/Model.out'
        #                                         if offset[j] == -0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C452_W8_T5.6_P0.0_10D_L-0.75/Model.out'
        #                                         if offset[j] == -0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C453_W8_T5.6_P0.0_10D_L-0.5/Model.out'
        #                                         if offset[j] == -0.25:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C454_W8_T5.6_P0.0_10D_L-0.25/Model.out'
        #                                         if offset[j] == 0.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C455_W8_T5.6_P0.0_10D_L0/Model.out'
        #                                         if offset[j] == 0.25:
        #                                                 filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C456_W8_T5.6_P0.0_10D_L0.25/Model.out'
        #                                         if offset[j] == 0.5:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C457_W8_T5.6_P0.0_10D_L0.5/Model.out'
        #                                         if offset[j] == 0.75:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C458_W8_T5.6_P0.0_10D_L0.75/Model.out'
        #                                         if offset[j] == 1.:
        #                                                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C459_W8_T5.6_P0.0_10D_L1.0/Model.out'
        #
        #                 lines = np.loadtxt(filename,skiprows=8)
        #                 time = lines[:,0]
        #                 my = lines[:,12]
        #                 time = time-time[0]
        #                 m_func = interp1d(time, my, kind='cubic')
        #
        #                 t = np.linspace(0.,600.,24000)
        #                 FAST_mom = np.zeros_like(t)
        #                 for k in range(len(t)-1):
        #                         FAST_mom[k] = m_func(t[k])
        #
        #                 damage[j] = calc_damage(FAST_mom,1.)
        #                 print damage[j]
        #
        #         print 'separation: ', sep[i]
        #         print 'damage: ', repr(damage)



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
        # N=24001
        # # #
        # TI = 0.11
        # print 'setup'
        # s = Time.time()
        # atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed = setup_atm(filename_free,filename_close,filename_far,TI=TI,N=N)
        # print Time.time()-s
        #
        # sep = np.array([4.,7.,10.])
        # # sep = np.array([7.])
        # offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
        #
        # start_time = Time.time()
        # for i in range(len(sep)):
        #         damage = np.zeros_like(offset)
        #         for j in range(len(offset)):
        #                 # print 'separation: ', sep[i]
        #                 # print 'offset: ', offset[j]
        #                 turbineX = np.array([0.,126.4*sep[i]])
        #                 turbineY = np.array([0.,126.4])*offset[j]
        #                 # windDirections = np.array([270.,270.])
        #                 # windFrequencies = np.array([0.25,0.75])
        #                 windDirections = np.array([270.])
        #                 windFrequencies = np.array([1.])
        #                 s = Time.time()
        #                 damage[j] = farm_damage(turbineX,turbineY,windDirections,windFrequencies,atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed,TI=TI,N=N)[1]
        #                 # print "iteration time: ", (Time.time()-s)/2.
        #                 print damage[j]
        #         print 'separation: ', sep[i]
        #         print 'damage: ', repr(damage)
        # print "time to run: ", (Time.time()-start_time)





        # NEW
        # TI = 0.11
        # print 'setup'
        # atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed = new_setup_atm(filename_free,filename_close,filename_far,TI=TI)
        #
        # sep = np.array([4.,7.,10.])
        # offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
        #
        # # start_time = Time.time()
        # for i in range(len(sep)):
        #         damage = np.zeros_like(offset)
        #         for j in range(len(offset)):
        #                 turbineX = np.array([0.,126.4*sep[i]])
        #                 turbineY = np.array([0.,126.4])*offset[j]
        #                 windDirections = np.array([270.])
        #                 windFrequencies = np.array([1.])
        #                 s = Time.time()
        #                 damage[j] = new_farm_damage(turbineX,turbineY,windDirections,windFrequencies,atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed,TI=TI)[1]
        #                 # print "iteration time: ", (Time.time()-s)/2.
        #                 # print damage[j]
        #         print 'separation: ', sep[i]
        #         print 'damage: ', repr(damage)
        # print "time to run: ", (Time.time()-start_time)




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





        #new 1.75
        # super4 = np.array([0.75584427, 0.75389668, 0.83245292, 0.93073563, 0.92723214,
        #         0.93439422, 0.84594801, 0.78022007, 0.75463485])
        # super7 = np.array([0.75323039, 0.78337442, 0.82640271, 0.81987042, 0.820295  ,
        #         0.84070687, 0.8211493 , 0.75954435, 0.7344193 ])
        # super10 = np.array([0.80175257, 0.85030926, 0.85032304, 0.83593441, 0.82197978,
        #         0.85018928, 0.84009284, 0.83577357, 0.79735362])

        #new 2.0
        # super4 = np.array([0.7508304 , 0.7909488 , 0.89013751, 0.93683538, 0.92723214,
        #         0.93743127, 0.87152195, 0.80970668, 0.7571569 ])
        # super7 = np.array([0.76355366, 0.80616546, 0.82766078, 0.81987042, 0.820295  ,
        #         0.84070687, 0.82402712, 0.79654953, 0.749165  ])
        # super10 = np.array([0.8294367 , 0.85109742, 0.85032304, 0.83593441, 0.82197978,
        #         0.85018928, 0.84009284, 0.83740819, 0.83605232])

        #new 2.0 weighted
        # super4 = np.array([0.74416874, 0.72968408, 0.78329996, 0.84586782, 0.86666946,
        #         0.83882614, 0.77934562, 0.75093583, 0.74989001])
        #
        # super7 = np.array([0.72992648, 0.74464389, 0.73719729, 0.74150187, 0.75063871,
        #         0.7555833 , 0.76034551, 0.71735714, 0.71463427])
        #
        # super10 = np.array([0.72717199, 0.70787166, 0.71860526, 0.72953176, 0.71331301,
        #         0.72421485, 0.7199894 , 0.71974853, 0.70593124])

        #new 3.0
        # super4 = np.array([0.87670289, 0.94503291, 0.93402053, 0.93683538, 0.92723214,
        #         0.93743127, 0.91765901, 0.92797553, 0.8668359 ])
        # super7 = np.array([0.82821056, 0.82584297, 0.82766078, 0.81987042, 0.820295  ,
        #         0.84070687, 0.82402712, 0.82133596, 0.83666382])
        # super10  =np.array([0.83769879, 0.85109742, 0.85032304, 0.83593441, 0.82197978,
        #         0.85018928, 0.84009284, 0.83740819, 0.85157443])







        #new new
       #  super4 = np.array([0.67277673, 0.76761523, 0.88343507, 0.87900442, 1.61663901,
       # 0.87865917, 0.88284892, 0.76717   , 0.67277212])
       #  super7 = np.array([0.71494113, 0.78593984, 0.79127318, 0.78575981, 1.32803624,
       # 0.78573813, 0.79119028, 0.78570273, 0.71473823])
       #  super10 = np.array([0.78196385, 0.77395874, 0.77272644, 0.76999199, 0.76690304,
       # 0.76998917, 0.77271769, 0.77395103, 0.78176563])
       #  freestreamCC = np.ones_like(FAST4)*0.62968317


       #  super4 = np.array([0.67277673, 0.76761523, 0.88343507, 0.87900442, 0.82745061,
       # 0.87865917, 0.88284892, 0.76717   , 0.67277212])
       #  super7 = np.array([0.71494113, 0.78593984, 0.79127318, 0.78575981, 0.74599994,
       # 0.78573813, 0.79119028, 0.78570273, 0.71473823])
       #  super10 = np.array([0.78196385, 0.77395874, 0.77272644, 0.76999199, 0.73339604,
       # 0.76998917, 0.77271769, 0.77395103, 0.78176563])

       #  super4 = np.array([0.7318122 , 0.80957443, 0.91593266, 1.00382115, 1.0227498 ,
       # 1.0020521 , 0.90209015, 0.80980546, 0.72887928])
       #  super7 = np.array([0.85057647, 0.93904326, 0.98096368, 0.98465143, 0.95989078,
       # 0.98838351, 0.96869113, 0.92639066, 0.83547668])
       #  super10 = np.array([0.89958267, 0.93595487, 0.94413618, 0.92833491, 0.89210531,
       # 0.93120719, 0.92008385, 0.91197694, 0.90614781])

       #2.0
      #   super4 = np.array([0.71531298, 0.83092194, 0.95409818, 1.01290402, 1.01290402,
      #  1.01290402, 0.95409818, 0.83092194, 0.71531298])
      #   super7 = np.array([0.81251641, 0.91225896, 0.94809132, 0.94809132, 0.94809132,
      # 0.94809132, 0.94809132, 0.91225896, 0.81251641])
      #   super10 = np.array([0.86256839, 0.88327863, 0.88327863, 0.88327863, 0.88327863,
      #  0.88327863, 0.88327863, 0.88327863, 0.86256839])

       #1.0
       #  super4 = np.array([0.61927481, 0.648916  , 0.73111831, 0.83236387, 0.89202716,
       # 0.83236387, 0.73111831, 0.648916  , 0.61927481])
       #  super7 = np.array([0.62429425, 0.68400553, 0.77196834, 0.87352005, 0.94809132,
       # 0.87352005, 0.77196834, 0.68400553, 0.62429425])
       #  super10 = np.array([0.64710586, 0.71277139, 0.79340773, 0.87266479, 0.88327863,
       # 0.87266479, 0.79340773, 0.71277139, 0.64710586])

        #1.75
       #  super4 = np.array([0.67356624, 0.77695019, 0.89938595, 1.00926579, 1.01290402,
       # 1.00926579, 0.89938595, 0.77695019, 0.67356624])
       #  super7 = np.array([0.75322667, 0.85652751, 0.94390874, 0.94809132, 0.94809132,
       # 0.94809132, 0.94390874, 0.85652751, 0.75322667])
       #  super10 = np.array([0.80977114, 0.87877124, 0.88327863, 0.88327863, 0.88327863,
       # 0.88327863, 0.88327863, 0.87877124, 0.80977114])


        super4 = np.array([0.63353188, 0.63493216, 0.70184102, 0.8257424 , 0.90239621,
       0.83189076, 0.67499081, 0.62958289, 0.637472  ])
        super7 = np.array([0.66408119, 0.72975469, 0.79053706, 0.91087253, 0.95989078,
       0.91353201, 0.80373111, 0.70644025, 0.64850947])
        super10 = np.array([0.69413751, 0.75912991, 0.84336117, 0.91783536, 0.89210531,
       0.92020988, 0.83027765, 0.75414277, 0.68091793])


        fig = plt.figure(figsize=[6.5,2.5])
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

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
        # ax1.plot(offset,FAST7,'o',label='SOWFA+FAST',color='C3',markersize=3)
        # ax1.plot(offset,super7,'o',label='superimposed',color='C0',markersize=3)
        # ax1.plot(offset,FASTfree,'ok',label='freestream FAST')
        # ax1.plot(offset,freestreamCC,'og',label='freestream CC')
        ax1.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black',label='freestream loads')
        # ax1.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0',label='freestream superimposed')
        # ax1.legend(loc=3,prop={'family':'serif', 'size':8})

        ax1.set_title('7D',family='serif',fontsize=8)
        ax1.set_ylabel('damage',family='serif',fontsize=8)

        ax1.set_xlabel('offset (D)',family='serif',fontsize=8)

        ax2.plot(offset,FAST7,'or',color='C3')
        ax2.plot(offset,super7,'ob',color='C0')
        # ax2.plot(offset,FASTfree,'ok')
        # ax2.plot(offset,freestreamCC,'og')
        ax2.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
        # ax2.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0')
        ax2.set_title('7D',family='serif',fontsize=8)
        ax2.set_xlabel('offset (D)',family='serif',fontsize=8)
        #
        ax3.plot(offset,FAST10,'or',color='C3')
        ax3.plot(offset,super10,'ob',color='C0')
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

        ax1.set_ylim(0.6,1.1)
        ax2.set_ylim(0.6,1.1)
        ax3.set_ylim(0.6,1.1)

        # plt.suptitle('1.5')
        # plt.savefig('fatigue_damage7D.pdf',transparent=True)
        plt.show()
