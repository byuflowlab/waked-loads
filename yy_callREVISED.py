
import numpy as np
import matplotlib.pyplot as plt
import time as Time
from yy_calc_fatigue import *
import sys
sys.dont_write_bytecode = True


if __name__ == '__main__':
      # T11
      # filename_free = '/home/flowlab/PJ/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
      # filename_close = '/home/flowlab/PJ/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
      # filename_far = '/home/flowlab/PJ/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
      # T11
      filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
      filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
      filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
      # T5.6
      # filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C464_W8_T5.6_P0.0_m2D_L0/Model.out'
      # filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C437_W8_T5.6_P0.0_4D_L0/Model.out'
      # filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C455_W8_T5.6_P0.0_10D_L0/Model.out'

      TI = 0.11

      ofree,sfree,oclose,sclose,ofar,sfar = find_omega(filename_free,filename_close,filename_far,TI=TI)
      Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()

      # windDirections = np.array([0.,270.])
      # windFrequencies = np.array([0.5,0.5])

      # turbineX = np.array([-30.189328192323742, 409.68557222509094, -325.29298851626515, 388.45347485863886, -459.1542760702638, -5.918263895454076, -318.709746908058, 0.01591293670720216, -122.72621666510742, 298.43484510880074])
      # turbineY = np.array([-135.9655851458904, 213.7063826527904, 327.87140365822717, -249.52881347955298, 54.75509676733264, 461.6863409631553, -334.05347901927723, -461.6863409631554, 176.62807682134303, -13.2990667664626])
      #
      #
      # damage = farm_damage(turbineX,turbineY,windDirections,windFrequencies,ofree,sfree,oclose,sclose,ofar,sfar,
      #                         Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI)


      windDirections = np.array([270.])
      windFrequencies = np.array([1.])

      diameter = 126.4
      x = np.array([0.,4.*diameter])
      offset = np.linspace(-3.,3.,25)
      damage = np.zeros_like(offset)
      for i in range(len(offset)):
            y = np.array([0.,offset[i]*diameter])
            damage[i] = farm_damage(x,y,windDirections,windFrequencies,ofree,sfree,oclose,sclose,ofar,sfar,
                                    Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI)[1]

      print damage
