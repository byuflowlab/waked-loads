
import numpy as np
import matplotlib.pyplot as plt
import time
# from zz_calc_fatigue import *
# from calc_fatigue_NAWEA import *
from yy_calc_fatigue import *
# import damage_calc


if __name__ == '__main__':

      sep = np.array([4.,7.,10.])
      offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
      # sep = np.array([4.])
      # offset = np.array([0.5])
      TI = 11.
      freestream = False
      for i in range(len(sep)):
            damage = np.zeros_like(offset)
            di = np.zeros_like(offset)
            for j in range(len(offset)):
                  if freestream == True:
                          if TI == 11.:
                                  # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C676_W8_T11.0_P0.0_m2D_L-1.0/Model.out'
                          if TI == 5.6:
                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C464_W8_T5.6_P0.0_m2D_L0/Model.out'

                  else:
                          if TI == 11.:
                                  if sep[i] == 4.:
                                          if offset[j] == -1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C649_W8_T11.0_P0.0_4D_L-1.0/Model.out'
                                          if offset[j] == -0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C650_W8_T11.0_P0.0_4D_L-0.75/Model.out'
                                          if offset[j] == -0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C651_W8_T11.0_P0.0_4D_L-0.5/Model.out'
                                          if offset[j] == -0.25:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C652_W8_T11.0_P0.0_4D_L-0.25/Model.out'
                                          if offset[j] == 0.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
                                          if offset[j] == 0.25:
                                                  filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C654_W8_T11.0_P0.0_4D_L0.25/Model.out'
                                          if offset[j] == 0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C655_W8_T11.0_P0.0_4D_L0.5/Model.out'
                                          if offset[j] == 0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C656_W8_T11.0_P0.0_4D_L0.75/Model.out'
                                          if offset[j] == 1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C657_W8_T11.0_P0.0_4D_L1.0/Model.out'
                                  if sep[i] == 7.:
                                          if offset[j] == -1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C658_W8_T11.0_P0.0_7D_L-1.0/Model.out'
                                          if offset[j] == -0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C659_W8_T11.0_P0.0_7D_L-0.75/Model.out'
                                          if offset[j] == -0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C660_W8_T11.0_P0.0_7D_L-0.5/Model.out'
                                          if offset[j] == -0.25:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C661_W8_T11.0_P0.0_7D_L-0.25/Model.out'
                                          if offset[j] == 0.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C662_W8_T11.0_P0.0_7D_L0/Model.out'
                                          if offset[j] == 0.25:
                                                  filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C663_W8_T11.0_P0.0_7D_L0.25/Model.out'
                                          if offset[j] == 0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
                                          if offset[j] == 0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C665_W8_T11.0_P0.0_7D_L0.75/Model.out'
                                          if offset[j] == 1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C666_W8_T11.0_P0.0_7D_L1.0/Model.out'
                                  if sep[i] == 10.:
                                          if offset[j] == -1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C667_W8_T11.0_P0.0_10D_L-1.0/Model.out'
                                          if offset[j] == -0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C668_W8_T11.0_P0.0_10D_L-0.75/Model.out'
                                          if offset[j] == -0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C669_W8_T11.0_P0.0_10D_L-0.5/Model.out'
                                          if offset[j] == -0.25:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C670_W8_T11.0_P0.0_10D_L-0.25/Model.out'
                                          if offset[j] == 0.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
                                          if offset[j] == 0.25:
                                                  filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C672_W8_T11.0_P0.0_10D_L0.25/Model.out'
                                          if offset[j] == 0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C673_W8_T11.0_P0.0_10D_L0.5/Model.out'
                                          if offset[j] == 0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C674_W8_T11.0_P0.0_10D_L0.75/Model.out'
                                          if offset[j] == 1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C675_W8_T11.0_P0.0_10D_L1.0/Model.out'

                          if TI == 5.6:
                                  if sep[i] == 4.:
                                          if offset[j] == -1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C433_W8_T5.6_P0.0_4D_L-1.0/Model.out'
                                          if offset[j] == -0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C434_W8_T5.6_P0.0_4D_L-0.75/Model.out'
                                          if offset[j] == -0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C435_W8_T5.6_P0.0_4D_L-0.5/Model.out'
                                          if offset[j] == -0.25:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C436_W8_T5.6_P0.0_4D_L-0.25/Model.out'
                                          if offset[j] == 0.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C437_W8_T5.6_P0.0_4D_L0/Model.out'
                                          if offset[j] == 0.25:
                                                  filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C438_W8_T5.6_P0.0_4D_L0.25/Model.out'
                                          if offset[j] == 0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C439_W8_T5.6_P0.0_4D_L0.5/Model.out'
                                          if offset[j] == 0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C440_W8_T5.6_P0.0_4D_L0.75/Model.out'
                                          if offset[j] == 1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C441_W8_T5.6_P0.0_4D_L1.0/Model.out'
                                  if sep[i] == 7.:
                                          if offset[j] == -1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C442_W8_T5.6_P0.0_7D_L-1.0/Model.out'
                                          if offset[j] == -0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C443_W8_T5.6_P0.0_7D_L-0.75/Model.out'
                                          if offset[j] == -0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C444_W8_T5.6_P0.0_7D_L-0.5/Model.out'
                                          if offset[j] == -0.25:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C445_W8_T5.6_P0.0_7D_L-0.25/Model.out'
                                          if offset[j] == 0.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C446_W8_T5.6_P0.0_7D_L0/Model.out'
                                          if offset[j] == 0.25:
                                                  filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C447_W8_T5.6_P0.0_7D_L0.25/Model.out'
                                          if offset[j] == 0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C448_W8_T5.6_P0.0_7D_L0.5/Model.out'
                                          if offset[j] == 0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C449_W8_T5.6_P0.0_7D_L0.75/Model.out'
                                          if offset[j] == 1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C450_W8_T5.6_P0.0_7D_L1.0/Model.out'
                                  if sep[i] == 10.:
                                          if offset[j] == -1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C451_W8_T5.6_P0.0_10D_L-1.0/Model.out'
                                          if offset[j] == -0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C452_W8_T5.6_P0.0_10D_L-0.75/Model.out'
                                          if offset[j] == -0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C453_W8_T5.6_P0.0_10D_L-0.5/Model.out'
                                          if offset[j] == -0.25:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C454_W8_T5.6_P0.0_10D_L-0.25/Model.out'
                                          if offset[j] == 0.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C455_W8_T5.6_P0.0_10D_L0/Model.out'
                                          if offset[j] == 0.25:
                                                  filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C456_W8_T5.6_P0.0_10D_L0.25/Model.out'
                                          if offset[j] == 0.5:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C457_W8_T5.6_P0.0_10D_L0.5/Model.out'
                                          if offset[j] == 0.75:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C458_W8_T5.6_P0.0_10D_L0.75/Model.out'
                                          if offset[j] == 1.:
                                                  filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C459_W8_T5.6_P0.0_10D_L1.0/Model.out'

                  lines = np.loadtxt(filename,skiprows=8)
                  time = lines[:,0]
                  my = lines[:,12]
                  mx = lines[:,11]

                  # plt.plot(my)
                  # plt.plot(mx)
                  # plt.show()

                  # mx = np.zeros_like(mx)
                  # plt.plot(mx[0:5000])
                  # print offset[j]
                  # print np.max(mx)-np.min(mx)
                  di[j] = np.max(mx)-np.min(mx)
                  # plt.show()

                  damage[j] = calc_damage_moments(mx,1.0,fos=2.)
                  # damage[j] = get_damage(my,1.0,fos=2.0)
                  # new_calc_damage(my,mx,1.0)
                  print damage[j]
                  # plt.figure(1)
                  # plt.plot(offset[j],1.2885465608898776,'or')

            # plt.figure(1)
            plt.plot(offset,damage,'o')
            # plt.figure(2)
            # plt.plot(offset,di,'o')
            print 'separation: ', sep[i]
            print 'damage: ', repr(damage)
      # plt.ylim(0.25,1.1)
      plt.show()





      # sep = 10.
      # offset = 0.
      #
      # if sep == 4.:
      #       if offset == -1.:
      #             filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C649_W8_T11.0_P0.0_4D_L-1.0/Model.out'
      #       if offset == -0.75:
      #             filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C650_W8_T11.0_P0.0_4D_L-0.75/Model.out'
      #       if offset == -0.5:
      #             filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C651_W8_T11.0_P0.0_4D_L-0.5/Model.out'
      #       if offset == -0.25:
      #             filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C652_W8_T11.0_P0.0_4D_L-0.25/Model.out'
      #       if offset == 0.:
      #             filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
      #       if offset == 0.25:
      #             filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C654_W8_T11.0_P0.0_4D_L0.25/Model.out'
      #       if offset == 0.5:
      #             filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C655_W8_T11.0_P0.0_4D_L0.5/Model.out'
      #       if offset == 0.75:
      #             filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C656_W8_T11.0_P0.0_4D_L0.75/Model.out'
      #       if offset == 1.:
      #             filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C657_W8_T11.0_P0.0_4D_L1.0/Model.out'
      # if sep[i] == 7.:
      #       if offset[j] == -1.:
      #               filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C658_W8_T11.0_P0.0_7D_L-1.0/Model.out'
      #       if offset[j] == -0.75:
      #               filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C659_W8_T11.0_P0.0_7D_L-0.75/Model.out'
      #       if offset[j] == -0.5:
      #               filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C660_W8_T11.0_P0.0_7D_L-0.5/Model.out'
      #       if offset[j] == -0.25:
      #               filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C661_W8_T11.0_P0.0_7D_L-0.25/Model.out'
      #       if offset[j] == 0.:
      #               filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C662_W8_T11.0_P0.0_7D_L0/Model.out'
      #       if offset[j] == 0.25:
      #               filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C663_W8_T11.0_P0.0_7D_L0.25/Model.out'
      #       if offset[j] == 0.5:
      #               filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
      #       if offset[j] == 0.75:
      #               filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C665_W8_T11.0_P0.0_7D_L0.75/Model.out'
      #       if offset[j] == 1.:
      #               filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C666_W8_T11.0_P0.0_7D_L1.0/Model.out'
      # if sep == 10.:
      #         if offset == -1.:
      #                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C667_W8_T11.0_P0.0_10D_L-1.0/Model.out'
      #         if offset == -0.75:
      #                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C668_W8_T11.0_P0.0_10D_L-0.75/Model.out'
      #         if offset == -0.5:
      #                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C669_W8_T11.0_P0.0_10D_L-0.5/Model.out'
      #         if offset == -0.25:
      #                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C670_W8_T11.0_P0.0_10D_L-0.25/Model.out'
      #         if offset == 0.:
      #                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
      #         if offset == 0.25:
      #                 filename  = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C672_W8_T11.0_P0.0_10D_L0.25/Model.out'
      #         if offset == 0.5:
      #                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C673_W8_T11.0_P0.0_10D_L0.5/Model.out'
      #         if offset == 0.75:
      #                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C674_W8_T11.0_P0.0_10D_L0.75/Model.out'
      #         if offset == 1.:
      #                 filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C675_W8_T11.0_P0.0_10D_L1.0/Model.out'
      #
      #
      #
      # lines = np.loadtxt(filename,skiprows=8)
      # time = lines[:,0]
      # my = lines[:,12]
      # mx = lines[:,11]
      #
      # plt.figure(1)
      # plt.plot(my[0:2000])
      #
      # plt.figure(2)
      # plt.plot(mx[0:2000])
      #
      # plt.show()
