
import numpy as np
import matplotlib.pyplot as plt
import time as Time
from zz_calc_fatigue import *
from calc_fatigue_NAWEA import farm_damage as FARM_DAMAGE
from calc_fatigue_NAWEA import extract_damage

if __name__ == '__main__':
      # T11
      filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
      filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
      filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
      # T5.6
      # filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C464_W8_T5.6_P0.0_m2D_L0/Model.out'
      # filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C437_W8_T5.6_P0.0_4D_L0/Model.out'
      # filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C455_W8_T5.6_P0.0_10D_L0/Model.out'

      damage_free,damage_close,damage_far = extract_damage(filename_free,filename_close,filename_far)


      sep = np.array([4.,7.,10.])
      offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])

      for i in range(len(sep)):
            damage = np.zeros_like(offset)
            for j in range(len(offset)):
                  turbineX = np.array([0.,126.4*sep[i]])
                  turbineY = np.array([0.,126.4])*offset[j]
                  # windDirections = np.array([270.,270.])
                  # windFrequencies = np.array([0.25,0.75])
                  windDirections = np.array([270.])
                  windFrequencies = np.array([1.])
                  damage[j] = FARM_DAMAGE(turbineX,turbineY,windDirections,windFrequencies,damage_free,damage_close,damage_far)[1]
            print 'separation: ', sep[i]
            print 'damage: ', repr(damage)



      # TI = 0.11
      #
      # start_setup = Time.time()
      # s = Time.time()
      # ffree, ofree, sfree, fclose, oclose, sclose, ffar, ofar, sfar = setup_atm_flap(filename_free,filename_close,filename_far,TI=TI)
      # print 'setup flap: ', Time.time()-s
      # s = Time.time()
      # upper_free,lower_free,upper_close,lower_close,upper_far,lower_far = setup_edge_bounds(filename_free,filename_close,filename_far,TI=TI)
      # print 'setup edge: ', Time.time()-s
      # s = Time.time()
      # recovery_dist = find_freestream_recovery(TI=TI)
      # print 'setup recovery: ', Time.time()-s
      # print 'setup time: ', Time.time()-start_setup
      #
      # turb_index = 1
      # sep = np.array([4.,7.,10.])
      # offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
      # for i in range(len(sep)):
      #       damage = np.zeros_like(offset)
      #       for j in range(len(offset)):
      #             turbineX = np.array([0.,126.4])*sep[i]
      #             turbineY = np.array([0.,126.4])*offset[j]
      #
      #             s = Time.time()
      #             flap,edge = get_loads_history(turbineX,turbineY,turb_index,ofree,oclose,ofar,sfree,sclose,
      #                       sfar,ffree,fclose,ffar,upper_free,lower_free,upper_close,lower_close,
      #                       upper_far,lower_far,recovery_dist,TI=TI)
      #             edge = np.zeros_like(flap)
      #             damage[j] = calc_damage_moments(flap,edge,1.0)
      #             print 'damage calc time: ', Time.time()-s
      #
      #             print damage[j]
      #
      #       print 'separation: ', sep[i]
      #       print 'damage: ', repr(damage)
