
import numpy as np
import matplotlib.pyplot as plt
import time as Time
from yy_calc_fatigue import *
import sys
sys.dont_write_bytecode = True


if __name__ == '__main__':
      # T11
      filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
      filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
      filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
      # T5.6
      # filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C464_W8_T5.6_P0.0_m2D_L0/Model.out'
      # filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C437_W8_T5.6_P0.0_4D_L0/Model.out'
      # filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C455_W8_T5.6_P0.0_10D_L0/Model.out'

      TI = 0.11

      start_setup = Time.time()
      s = Time.time()
      ofree,sfree,oclose,sclose,ofar,sfar = find_omega(filename_free,filename_close,filename_far,TI=TI)
      print 'find omega: ', Time.time()-s

      print ofree,sfree,oclose,sclose,ofar,sfar

      turb_index = 1
      sep = np.array([4.,7.,10.])
      offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])
      # offset = np.array([-3.0,-2.75,-2.5,-2.25,-2.,-1.75,-1.5,-1.25,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.])

      Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()
      print Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg

      # sep = np.array([4.])
      # offset = np.array([0.25])
      s = Time.time()
      for i in range(len(sep)):
            damage = np.zeros_like(offset)
            for j in range(len(offset)):
                  turbineX = np.array([0.,126.4])*sep[i]
                  turbineY = np.array([0.,126.4])*offset[j]
                  damage[j] = get_edgewise_damage(turbineX,turbineY,turb_index,ofree,sfree,oclose,sclose,ofar,sfar,
                                    Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI)

                  # print 'one: ', damage[j]
                  # windDirections = np.array([270.,270.,270.])
                  # windFrequencies = np.array([0.45,0.25,0.3])
                  # windDirections = np.array([270.])
                  # windFrequencies = np.array([1.])
                  # print 'two: ', farm_damage(turbineX,turbineY,windDirections,windFrequencies,ofree,sfree,oclose,sclose,ofar,sfar,
                  #                         Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI)

            print 'separation: ', sep[i]
            print 'damage: ', repr(damage)

      print (Time.time()-s)/(len(sep)*len(offset))
