
import numpy as np
import matplotlib.pyplot as plt
import time
from zz_flapwise_model import *
import damage_calc


if __name__ == '__main__':
      #
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

      print 'setup'
      s = time.time()
      damage_free,damage_close,damage_far,recovery_dist = preprocess_damage(filename_free,filename_close,filename_far)
      print 'setup time: ', time.time()-s
      print 'damage_free: ', damage_free
      print 'damage_close ', damage_close
      print 'damage_far: ', damage_far
      print 'recovery_dist: ', recovery_dist
