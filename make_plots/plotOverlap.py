import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from math import sin, cos, radians

if __name__=='__main__':

    # import matplotlib as mpl
    # mpl.rc('font',family='serif')
    fig = plt.figure(figsize=[2.5,2.])
    ax1 = plt.gca()

    ax1.axis('off')


    """variable heights"""
    H1 = 90.
    d1 = np.array([6.3, 4.5, 3.8])
    D1 = 126.4

    R1 = D1/2.

    spacing = 0.8
    r = 126.4/2.
    x1 = 1.5*r
    circle1 = plt.Circle((x1,H1), R1, color='black', fill=False, linestyle = '--', linewidth=1.)

    ax1.add_artist(circle1)

    c1 = R1/35.

    px1 = np.array([x1-d1[0]/2,x1-d1[1]/2,x1-d1[2]/2,x1+d1[2]/2,x1+d1[1]/2,x1+d1[0]/2,x1-d1[0]/2])
    py1 = np.array([0,H1/2,H1-3.*c1,H1-3.*c1,H1/2,0,0])
    ax1.plot(px1,py1,color='black', linewidth=1.)


    #add blades
    hub1 = plt.Circle((x1,H1), 3*c1, color='black', fill=False, linewidth=1)
    ax1.add_artist(hub1)
    bladeX = np.array([3.,7.,10.,15.,20.,25.,30.,35.,30.,25.,20.,15.,10.,5.,3.,3.])
    bladeY = (np.array([0.,0.,0.8,1.5,1.7,1.9,2.1,2.3,2.4,2.4,2.4,2.4,2.4,2.4,2.4,0.])-1.5)*1.5

    angle1 = np.random.rand(1)*60.-55.

    blade1X = bladeX*cos(radians(angle1))-bladeY*sin(radians(angle1))
    blade1Y = bladeX*sin(radians(angle1))+bladeY*cos(radians(angle1))

    blade2X = bladeX*cos(radians(angle1+120.))-bladeY*sin(radians(angle1+120.))
    blade2Y = bladeX*sin(radians(angle1+120.))+bladeY*cos(radians(angle1+120.))

    blade3X = bladeX*cos(radians(angle1+240.))-bladeY*sin(radians(angle1+240.))
    blade3Y = bladeX*sin(radians(angle1+240.))+bladeY*cos(radians(angle1+240.))

    ax1.plot(blade1X*c1+x1, blade1Y*c1+H1, linewidth=1, color='black')
    ax1.plot(blade2X*c1+x1, blade2Y*c1+H1, linewidth=1, color='black')
    ax1.plot(blade3X*c1+x1, blade3Y*c1+H1, linewidth=1, color='black')


    ax1.axis('equal')

    ax1.set_xlim([0.5*r,200.])
    ax1.set_ylim([-1.,150])


    # ax1.text(150.,180.,'mixed wind farm heights',horizontalalignment='center')


    # plt.savefig('/Users/ningrsrch/Dropbox/Projects/collaborate/proposal-vpm-wind-farm/figures/mixed-heights.pdf',transparent=True)
    plt.show()
