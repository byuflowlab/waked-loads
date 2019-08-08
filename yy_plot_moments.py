
import numpy as np
import matplotlib.pyplot as plt
from yy_calc_fatigue import *


if __name__ == '__main__':

        fig = plt.figure(1,figsize=[5.,2.5])
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C673_W8_T11.0_P0.0_10D_L0.5/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C655_W8_T11.0_P0.0_4D_L0.5/Model.out'

        lines = np.loadtxt(filename,skiprows=8)
        angles = lines[:,5]
        mx_FAST = lines[:,11]

        # angles = angles[0:1000]
        # mx_FAST = mx_FAST[0:1000]

        ang = np.array([])
        mom = np.array([])
        for i in range(1000):
                ang = np.append(ang,angles[i+37])
                mom = np.append(mom,mx_FAST[i+37])
                if angles[i+38] < angles[i+37]:
                        ax1.plot(ang,mom,color='C0')
                        ang = np.array([])
                        mom = np.array([])

        Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()

        hub_height = 90.

        # angles = np.linspace(0.,720.,100)
        edge1 = np.zeros_like(angles)
        flap1 = np.zeros_like(angles)

        edge2 = np.zeros_like(angles)
        flap2 = np.zeros_like(angles)

        turbineX = np.array([0.,126.4])*10.

        turbineY1 = np.array([0.,126.4])*0.5

        angles = np.linspace(0.,360.,100)
        edge1 = np.zeros_like(angles)
        flap1 = np.zeros_like(angles)
        for i in range(len(angles)):

                az = angles[i]
                #freestream
                x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY1[1],hub_height,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX, turbineY1, x_locs, y_locs, z_locs, 8.0,TI=0.11)
                flap1[i], edge1[i] = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)

        ax1.plot(angles,edge1/1000.,color='C1',linewidth=2)



        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C664_W8_T11.0_P0.0_7D_L0.5/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
        # filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'
        filename = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C651_W8_T11.0_P0.0_4D_L-0.5/Model.out'

        lines = np.loadtxt(filename,skiprows=8)
        angles = lines[:,5]
        mx_FAST = lines[:,11]

        # angles = angles[0:1000]
        # mx_FAST = mx_FAST[0:1000]

        ang = np.array([])
        mom = np.array([])
        for i in range(1000):
                ang = np.append(ang,angles[i+37])
                mom = np.append(mom,mx_FAST[i+37])
                if angles[i+38] < angles[i+37]:
                        ax2.plot(ang,mom,color='C0')
                        ang = np.array([])
                        mom = np.array([])

        Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()

        hub_height = 90.

        # angles = np.linspace(0.,720.,100)
        edge1 = np.zeros_like(angles)
        flap1 = np.zeros_like(angles)

        edge2 = np.zeros_like(angles)
        flap2 = np.zeros_like(angles)

        turbineX = np.array([0.,126.4])*4.

        turbineY1 = np.array([0.,126.4])*-0.5

        angles = np.linspace(0.,360.,100)
        edge1 = np.zeros_like(angles)
        flap1 = np.zeros_like(angles)
        for i in range(len(angles)):

                az = angles[i]
                #freestream
                x_locs,y_locs,z_locs = findXYZ(turbineX[1],turbineY1[1],hub_height,r,yaw_deg,az)
                speeds, _ = get_speeds(turbineX, turbineY1, x_locs, y_locs, z_locs, 8.0,TI=0.11)
                flap1[i], edge1[i] = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,8.0,pitch,azimuth=az)

        ax2.plot(angles,edge1/1000.,color='C1',linewidth=2)



        ax1.set_xlim(0.,360.)
        ax1.set_ylim(-3700.,4300.)
        ax2.set_xlim(0.,360.)
        ax2.set_ylim(-3700.,4300.)
        plt.show()
