
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as Time
from calc_fatigue_exp import *
# from calc_fatigue import *


if __name__ == '__main__':

        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C658_W8_T11.0_P0.0_7D_L-1.0/Model.out'
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C659_W8_T11.0_P0.0_7D_L-0.75/Model.out'
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C660_W8_T11.0_P0.0_7D_L-0.5/Model.out'
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C661_W8_T11.0_P0.0_7D_L-0.25/Model.out'
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C662_W8_T11.0_P0.0_7D_L0/Model.out'
        # # filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C663_W8_T11.0_P0.0_7D_L0.25/Model.out'
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C665_W8_T11.0_P0.0_7D_L0.75/Model.out'
        # # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C666_W8_T11.0_P0.0_7D_L1.0/Model.out'
        # lines = np.loadtxt(filename,skiprows=8)
        # time = lines[:,0]
        # my = lines[:,12]
        # time = time-time[0]
        # m_func = interp1d(time, my, kind='linear')
        # # m_func = interp1d(time, my, kind='cubic')
        #
        # N = np.linspace(20000,50000,5)
        # print N
        # damage = np.ones_like(N)
        #
        # for i in range(len(N)):
        #         t = np.linspace(0.,600.,int(N[i]))
        #         FAST_mom = np.zeros_like(t)
        #         for k in range(len(t)):
        #                 FAST_mom[k] = m_func(t[k])
        #         # plt.plot(t,FAST_mom)
        #         # plt.show()
        #
        #         damage[i] = calc_damage(FAST_mom,1.)
        #
        #         print 'damage: ', damage[i]
        #
        # plt.show()
        # plt.plot(N,damage,'o')
        # plt.show()



        # sep = np.array([4.,7.,10.])
        # offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
        # # sep = np.array([10.])
        # # offset = np.array([0.])
        # TI = 11.
        # freestream = True
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


        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C655_W8_T11.0_P0.0_4D_L0.5/Model.out'
        # filename='/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        # lines = np.loadtxt(filename,skiprows=8)
        # time = lines[:,0]
        # time = time-time[0]
        # my = lines[:,12]
        #
        # fig = plt.figure(figsize=[2.,2.])
        # plt.plot(time,my,linewidth=0.3)
        #
        # import matplotlib as mpl
        # label_size = 6
        # mpl.rcParams['xtick.labelsize'] = label_size
        # mpl.rcParams['ytick.labelsize'] = label_size
        #
        # fontProperties = {'family':'serif','size':8}
        # plt.gca().set_xticklabels(plt.gca().get_xticks(), fontProperties)
        # plt.gca().set_yticklabels(plt.gca().get_yticks(), fontProperties)
        #
        #
        # plt.gca().set_xticks((0.,300.,600.))
        # plt.gca().set_yticks((2000.,4000.,6000.))
        #
        # plt.gca().set_xticklabels(('0','5','10'))
        # plt.gca().set_yticklabels(('2','4','6'))
        #
        # plt.gca().set_xlabel('time (m)',family='serif',fontsize=8)
        # plt.gca().set_ylabel('root bending moment (MN-m)',family='serif',fontsize=8)
        #
        # plt.subplots_adjust(top = 0.9, bottom = 0.2, right = 0.98, left = 0.2,
        #             hspace = 0, wspace = 0.1)
        #
        # plt.savefig('moments-hist.pdf',transparent=True)
        # plt.show()


        # T11
        filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
        filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
        # T5.6
        # filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C464_W8_T5.6_P0.0_m2D_L0/Model.out'
        # filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C437_W8_T5.6_P0.0_4D_L0/Model.out'
        # filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C455_W8_T5.6_P0.0_10D_L0/Model.out'
        #
        # N=24001
        # #
        # TI = 0.11
        # print 'setup'
        # s = Time.time()
        # atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed = setup_atm(filename_free,filename_close,filename_far,TI=TI,N=N)
        # print Time.time()-s
        #
        # sep = np.array([4.,7.,10.])
        # offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
        #
        # # sep = np.array([4.])
        # # offset = np.array([0.5])
        # # offset = np.array([90000000.])
        # start_time = Time.time()
        # for i in range(len(sep)):
        #         damage = np.zeros_like(offset)
        #         for j in range(len(offset)):
        #                 turbineX = np.array([0.,126.4*sep[i]])
        #                 turbineY = np.array([0.,126.4])*offset[j]
        #                 windDirections = np.array([270.])
        #                 windFrequencies = np.array([1.])
        #                 s = Time.time()
        #                 damage[j] = farm_damage(turbineX,turbineY,windDirections,windFrequencies,atm_free,atm_close,atm_far,Omega_free,Omega_waked,free_speed,waked_speed,TI=TI,N=N)[1]
        #                 print "time to run: ", (Time.time()-s)/2.
        #                 print damage[j]
        #         # plt.show()
        #         print 'separation: ', sep[i]
        #         print 'damage: ', repr(damage)
        # print "time to run: ", (Time.time()-start_time)


        # plt.show()

        #superimposed

        offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
        #N=500
        # FAST4 = np.array([0.07894047, 0.09125097, 0.08585821, 0.08421438, 0.08078904, 0.08094466, 0.08223776, 0.08265324, 0.08035885])
        # FAST7 = np.array([0.08063268, 0.08051481, 0.0828863 , 0.08146759, 0.07791258, 0.07971275, 0.07499716, 0.07493915, 0.07593404])
        # FAST10 = np.array([0.0793749 , 0.07784707, 0.07665218, 0.0749598 , 0.07479351, 0.07672256, 0.07480289, 0.07575761, 0.08022163])
        #N=1000
        # FAST4 = np.array([0.13241257, 0.13588495, 0.13432356, 0.13835195, 0.14749183, 0.13781717, 0.12684397, 0.12813927, 0.13311428])
        # FAST7 = np.array([0.12511852, 0.12176602, 0.12156934, 0.12456163, 0.12654227, 0.13297344, 0.1239909,  0.12554152, 0.12478414])
        # FAST10 = np.array([0.12444908, 0.1203636,  0.12413859, 0.12560714, 0.12266063, 0.12287413,0.11863171, 0.12377686, 0.12591788])
        #N=5000
        # FAST4 = np.array([0.34001725, 0.32290031, 0.33306838, 0.32920632, 0.33911529, 0.34634675, 0.34485311, 0.34872447, 0.33984323])
        # FAST7 = array([0.32748083, 0.30784835, 0.32859986, 0.3191369 , 0.3249533 , 0.33199183, 0.33490999, 0.33909839, 0.33634221])
        # FAST10 = array([0.32770008, 0.33164743, 0.32139038, 0.32469743, 0.3267058 , 0.33054806, 0.33625501, 0.34104889, 0.3377739 ])


        #N=500
        # super4 = np.array([0.07948215, 0.08813901, 0.09340212, 0.07861026, 0.07829657, 0.08182726, 0.09708462, 0.08617302, 0.08020773])
        # super7 = np.array([0.07627224, 0.07784407, 0.07873344, 0.06846228, 0.07048317, 0.0727608, 0.07920184, 0.07918085, 0.07856352])
        # super10 = np.array([0.07882868, 0.08348917, 0.07860653, 0.07454765, 0.07579199, 0.07570652, 0.07614343, 0.08923353, 0.07835757])
        #N=1000
        # super4 =  np.array([0.13001833, 0.13420045, 0.14393586, 0.1286295 , 0.13599759, 0.12735207, 0.13761519, 0.13115964, 0.13239937])
        # super7 =  np.array([0.13168363, 0.12983282, 0.12554457, 0.11689232, 0.12355624, 0.12036148, 0.12348351, 0.12479704, 0.12747873])
        # super10 = np.array([0.13510289, 0.13881448, 0.12768255, 0.12214209, 0.12206444, 0.12281436, 0.11942748, 0.13859278, 0.13110133])
        #N=5000
        # super4 = array([0.14871786, 0.16205845, 0.1802348 , 0.1525766 , 0.15459722, 0.1548352 , 0.17312869, 0.15639837, 0.15050347])
        # super7 = np.array([0.15519635, 0.16522244, 0.14835871, 0.13787182, 0.13740996, 0.13875097, 0.1543059 , 0.15098156, 0.15135018])
        # super10 = np.array([0.15741442, 0.17720971, 0.14801908, 0.14191597, 0.14387224, 0.14221807, 0.14599026, 0.17276584, 0.15632148])

        #N=FULL DATA (1.5*wake)
        # super4 = np.array([0.63246622, 0.62637067, 0.7319896 , 0.98711861, 1.06574236, 0.97162182, 0.72040031, 0.62264971, 0.63726112])
        # super7 = np.array([0.6545269 , 0.71903671, 0.76980207, 0.94858193, 0.9677191 , 0.9673568 , 0.76591663, 0.69433639, 0.64363943])
        # super10 = np.array([0.66014054, 0.79942797, 0.87591593, 0.94864875, 0.95681256, 0.96344945, 0.8968796 , 0.76596455, 0.66884474])
        # #N=FULL (0.5 wake)
        # super4 = np.array([0.63246622, 0.60551304, 0.65888998, 0.74808353, 0.84929206, 0.74436326, 0.632559  , 0.60195723, 0.63726112])
        # super7 = np.array([0.6562383 , 0.67878037, 0.70503083, 0.7395229 , 0.79238848, 0.75585636, 0.71664028, 0.64906081, 0.64619133])
        # super10 = np.array([0.6615508 , 0.70405453, 0.72969373, 0.77879629, 0.79111886, 0.78445637, 0.73369202, 0.71277103, 0.66051297])
        #N=FULL (2.0 wake)
        # super4 = np.array([0.63246622, 0.63215356, 0.75000497, 1.03865597, 1.11004894, 1.04386814, 0.73348646, 0.62938972, 0.63726112])
        # super7 = np.array([0.65461076, 0.72330603, 0.77972846, 0.96885947, 0.98196078, 0.97809044, 0.76902704, 0.69145038, 0.63852347])
        # super10 = np.array([0.66546194, 0.81175455, 0.95397698, 1.00410749, 1.01207033, 1.00610536, 0.94437824, 0.79827915, 0.68349852])

        # #N=FULL DATA
        # FAST4 = np.array([0.77208607, 0.84301158, 0.98214149, 1.0404579 , 1.01968396, 1.05293275, 1.02057731, 0.9109089 , 0.7991055 ])
        # FAST7 = np.array([0.79800391, 0.85273074, 0.90235949, 0.9509322 , 1.00282632, 0.95212582, 0.92136755, 0.89915201, 0.84002526])
        # FAST10 = np.array([0.79923674, 0.88938653, 0.87021251, 0.86827795, 0.89101762, 0.91150866, 0.90512195, 0.85187077, 0.82026554])
        # #N=FULL DATA (takes a long time)
        # super4 = np.array([0.63246622, 0.61396962, 0.70366203, 0.89382563, 1.02211491, 0.87959704, 0.69953106, 0.60716793, 0.63726112])
        # super7 = np.array([0.6509985 , 0.70255895, 0.77522708, 0.90238161, 0.9782026 , 0.93213869, 0.78780946, 0.68852432, 0.64251256])
        # super10 = np.array([0.65977938, 0.77239248, 0.84824983, 0.91932477, 0.93476556, 0.92865426, 0.84277908, 0.74639778, 0.65672432])

        FAST4 = np.array([0.7479882 , 0.79418754, 0.91482836, 0.95437402, 0.9302589 ,0.96215701, 0.95952926, 0.86900191, 0.77281552])
        FAST7 = np.array([0.7479882 , 0.79418754, 0.91482836, 0.95437402, 0.9302589 ,0.96215701, 0.95952926, 0.86900191, 0.77281552])
        FAST10 = np.array([0.7664489 , 0.85394419, 0.82298162, 0.83205289, 0.84391592, 0.85639593, 0.86367084, 0.82085985, 0.78787999])
        # free = 0.62968317
        # FASTfree = np.ones_like(FAST4)*0.63117579
        FASTfree = np.ones_like(FAST4)*0.6373399523538535

        # super4 = np.array([0.52522855, 0.49643042, 0.55631978, 0.64730251, 0.7437337, 0.655517, 0.55025193, 0.48656372, 0.52323469])
        # super7 = np.array([0.52562318, 0.55708747, 0.59685104, 0.66869857, 0.72777291, 0.67676572, 0.59692333, 0.55486769, 0.5238179 ])
        # super10 = np.array([0.52256282, 0.5967802 , 0.64462104, 0.69469446, 0.70340349, 0.70559067, 0.62832159, 0.59447559, 0.52821153])

        #2* rad
        # super4 = np.array([0.67600654, 0.81147088, 0.94197874, 1.00382115, 1.0227498 , 1.0020521 , 0.94354412, 0.80575476, 0.7022973 ])
        # super7 = np.array([0.81066926, 0.96789703, 0.96769521, 0.9762949 , 0.95764617, 0.97673322, 0.95440725, 0.91732906, 0.81578342])
        # super10 = np.array([0.91137728, 0.93595487, 0.94413618, 0.92833491, 0.89210531, 0.93120719, 0.92008385, 0.91197694, 0.90520249])

        #1.75*rad
        # super4 = np.array([0.6465743 , 0.78860154, 0.90094437, 1.00180077, 1.0227498 , 1.00450329, 0.89131835, 0.77691189, 0.6600066 ])
        # super7 = np.array([0.77986687, 0.90528373, 0.96459293, 0.9762949 , 0.95764617, 0.97673322, 0.95207005, 0.83987861, 0.79218427])
        # super10 = np.array([0.84477878, 0.93358042, 0.94413618, 0.92833491, 0.89210531, 0.93120719, 0.92008385, 0.9078248 , 0.83510624])
       #  super4 = np.array([0.64407964, 0.77874452, 0.90250068, 1.00258973, 1.0227498 ,
       #                          1.00371455, 0.89010514, 0.77154326, 0.65945321])
       #  super7 = np.array([0.76890235, 0.90406207, 0.964592  , 0.9762949 , 0.95764617,
       #                          0.97673322, 0.95088826, 0.83786392, 0.78751689])
       #  super10 = np.array([0.8470996 , 0.93515408, 0.94413618, 0.92833491, 0.89210531,
       # 0.93120719, 0.92008385, 0.90821004, 0.83267762])


        #1.5*rad
        # super4 = np.array([0.63079717, 0.70790713, 0.82569682, 1.00033454, 1.0227498 , 0.97183418, 0.80606136, 0.7054441 , 0.64222151])
        # super7 = np.array([0.70187135, 0.81749768, 0.92617547, 0.9762949 , 0.95764617, 0.97673322, 0.93479795, 0.76797933, 0.71548929])
        # super10 = np.array([0.754343  , 0.90503389, 0.94413618, 0.92833491, 0.89210531, 0.93120719, 0.92008385, 0.85395571, 0.75010338])

        rms4 = np.sum(np.sqrt((FAST4-super4)**2))
        rms7 = np.sum(np.sqrt((FAST7-super7)**2))
        rms10 = np.sum(np.sqrt((FAST10-super10)**2))

        print 'rms4: ', rms4
        print 'rms7: ', rms7
        print 'rms10: ', rms10

        # freeCC = 0.6668796
        freestreamCC = np.ones_like(FAST4)*0.62968317
        #
        #
        #
        #
        # # TI=5.6
        #
        # FAST4 = np.array([0.62301502, 0.76051433, 0.99362264, 1.06089069, 0.97718426, 0.96806468, 0.96699219, 0.77118321, 0.64583782])
        # FAST7 = np.array([0.74998387, 0.83608668, 0.87085191, 0.91296148, 0.9531732, 0.97340949, 0.93045233, 0.85915521, 0.72402707])
        # FAST10 = np.array([0.83042745, 0.85503197, 0.86062394, 0.88745736, 0.89488398, 0.89238626, 0.9146126 , 0.89057379, 0.7829162])
        #
        # super4 = np.array([0.52217307, 0.42162574, 0.47914161, 0.64487447, 0.96960469, 0.64651494, 0.46757891, 0.4315811 , 0.49051586])
        # super7 = np.array([0.52630673, 0.44233444, 0.53781089, 0.73134672, 1.00773201, 0.75115251, 0.54007101, 0.4426996 , 0.51038692])
        # super10 = np.array([0.52407581, 0.51105356, 0.60483369, 0.86338228, 0.93715962, 0.84545417, 0.58249604, 0.53396496, 0.54593372])
        #
        # free = 0.51860783
        # freestream = np.ones_like(super4)*free
        #
        # freeCC = 0.58233807
        # freestreamCC = np.ones_like(super4)*freeCC


        #N=40000
        # TI = 11.0
        # FAST4 = np.array([0.93456824, 1.01442556, 1.18462055, 1.21717498, 1.21230796, 1.25260102, 1.21474347, 1.07548221, 0.95859563])
        # FAST7 = np.array([0.96755452, 1.03268292, 1.10318388, 1.1520673 , 1.20648398, 1.15165784, 1.11661191, 1.09052555, 1.01220717])
        # FAST10 = np.array([0.97389354, 1.06534524, 1.05747068, 1.06375754, 1.07243865, 1.09363711, 1.09966326, 1.01933611, 0.99143165])
        # FASTfree = np.ones_like(FAST4)*0.75885006












        # fig = plt.figure(figsize=[2.,2.])
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
        # ax1.plot(offset,FAST4,'o',label='SOWFA+FAST',color='C3')
        # ax1.plot(offset,super4,'o',label='superimposed',color='C0')
        ax1.plot(offset,FAST7,'o',label='SOWFA+FAST',color='C3')
        ax1.plot(offset,super7,'o',label='superimposed',color='C0')
        # ax1.plot(offset,FASTfree,'ok',label='freestream FAST')
        # ax1.plot(offset,freestreamCC,'og',label='freestream CC')
        ax1.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black',label='freestream loads')
        # ax1.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0',label='freestream superimposed')
        ax1.legend(loc=3,prop={'family':'serif', 'size':8})

        ax1.set_title('7D',family='serif',fontsize=8)
        ax1.set_ylabel('damage',family='serif',fontsize=8)

        ax2.set_xlabel('offset (D)',family='serif',fontsize=8)

        ax2.plot(offset,FAST7,'or',color='C3')
        ax2.plot(offset,super7,'ob',color='C0')
        # ax2.plot(offset,FASTfree,'ok')
        # ax2.plot(offset,freestreamCC,'og')
        ax2.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
        # ax2.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0')
        ax2.set_title('7D',family='serif',fontsize=8)
        ax2.set_xlabel('offset (D)',family='serif',fontsize=8)

        ax3.plot(offset,FAST10,'or',color='C3')
        ax3.plot(offset,super10,'ob',color='C0')
        # ax3.plot(offset,FASTfree,'ok')
        # ax3.plot(offset,freestreamCC,'og')
        ax3.set_title('10D',family='serif',fontsize=8)
        ax3.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
        # ax3.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0')

        ax1.set_ylim(0.,1.1)
        ax2.set_ylim(0.,1.1)
        ax3.set_ylim(0.,1.1)
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

        # plt.subplots_adjust(top = 0.9, bottom = 0.2, right = 0.98, left = 0.25,
        #             hspace = 0, wspace = 0.1)
        plt.subplots_adjust(top = 0.99, bottom = 0.2, right = 0.98, left = 0.1,
                    hspace = 0, wspace = 0.1)

        # plt.suptitle('1.5')
        plt.savefig('fatigue15.pdf',transparent=True)
        plt.show()
