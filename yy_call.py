
import numpy as np
import matplotlib.pyplot as plt
import time as Time
from calc_fatigue import *
import sys
sys.dont_write_bytecode = True


if __name__ == '__main__':

      TI = 0.11
      Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()

      windDirections = np.array([0.,270.])
      windFrequencies = np.array([0.5,0.5])

      turbineX = np.array([-30.189328192323742, 409.68557222509094, -325.29298851626515, 388.45347485863886, -459.1542760702638, -5.918263895454076, -318.709746908058, 0.01591293670720216, -122.72621666510742, 298.43484510880074])
      turbineY = np.array([-135.9655851458904, 213.7063826527904, 327.87140365822717, -249.52881347955298, 54.75509676733264, 461.6863409631553, -334.05347901927723, -461.6863409631554, 176.62807682134303, -13.2990667664626])

      print 'windDirections: ', windDirections
      print 'windFrequencies: ', windFrequencies
      print 'Rhub: ', Rhub
      print 'r: ', r
      print 'chord: ', chord
      print 'theta: ', theta
      print 'af: ', af
      print 'Rtip: ', Rtip
      print 'B: ', B
      print 'rho: ', rho
      print 'mu: ', mu
      print 'precone: ', precone
      print 'hubHt: ', hubHt
      print 'nSector: ', nSector
      print 'pitch: ', pitch
      print 'yaw_deg: ', yaw_deg
      print 'TI: ', TI

      damage = farm_damage(turbineX,turbineY,windDirections,windFrequencies,Rhub,r,chord,theta,af,
                                Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=0.11)

      print damage
