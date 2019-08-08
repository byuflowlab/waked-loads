import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from math import sin, cos, radians

if __name__=='__main__':

    # import matplotlib as mpl
    # mpl.rc('font',family='serif')
    fig = plt.figure(figsize=[5.5,6.])
    ax1 = plt.subplot(321)
    ax2 = plt.subplot(322)
    ax5 = plt.subplot(323)
    ax4 = plt.subplot(324)
    ax3 = plt.subplot(325)
    ax6 = plt.subplot(326)

    ax1.axis('off')
    ax3.axis('off')
    ax5.axis('off')

    import matplotlib as mpl
    mpl.rc('font', family = 'serif', serif = 'cmr10')

    H1 = 0.
    D1 = 126.4
    R1 = D1/2.
    spacing = 0.8
    r = 126.4/2.
    x1 = 100

    c1 = R1/35.

    #add blades
    hub1 = plt.Circle((x1,H1), 3*c1, color='black', fill=False, linewidth=1)
    ax1.add_artist(hub1)
    bladeX = np.array([3.,3.,5.,6.,10.,13.,18.,23.,28.,33.,38.,33.,28.,23.,18.,13.,7.,6.,3.])
    bladeY = -(np.array([2.,0.6,0.6,0.,0.,0.8,1.5,1.7,1.9,2.1,2.3,2.4,2.4,2.4,2.4,2.4,2.4,2.4,2.])-1.5)*1.5

    angle1 = 0.

    blade1X = bladeX*cos(radians(angle1))-bladeY*sin(radians(angle1))
    blade1Y = bladeX*sin(radians(angle1))+bladeY*cos(radians(angle1))

    blade2X = bladeX*cos(radians(angle1+180.))-bladeY*sin(radians(angle1+180.))
    blade2Y = bladeX*sin(radians(angle1+180.))+bladeY*cos(radians(angle1+180.))


    ax1.plot(blade1X*c1+x1, blade1Y*c1+H1, linewidth=1, color='black')
    ax1.plot(blade2X*c1+x1, blade2Y*c1+H1, linewidth=1, color='black')

    ax1.arrow(x1-30.,0.,0.,-30.,head_width=5.,head_length=5.,ec='C0',fc='C0')
    ax1.arrow(x1+30.,0.,0.,-30.,head_width=5.,head_length=5.,ec='C0',fc='C0')

    ax1.arrow(x1-40.,0.,0.,20.,head_width=5.,head_length=5.,ec='C1',fc='C1')
    ax1.arrow(x1+40.,0.,0.,-20.,head_width=5.,head_length=5.,ec='C1',fc='C1')

    ax1.text(x1-30.,-42.,'gravity',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='C0')
    ax1.text(x1-40.,32.,'aerodynamic',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='C1')

    tx = np.array([x1-3.,x1-3.,x1+3.,x1+3.])
    ty = np.array([H1-60.,H1-3.*c1,H1-3.*c1,H1-60.])
    ax1.plot(tx,ty,'-k')

    ax1.text(x1-r+5.,7.,'1',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax1.text(x1+r-5.,7.,'2',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')





    H1 = 0.
    D1 = 126.4
    R1 = D1/2.
    spacing = 0.8
    r = 126.4/2.
    x1 = r
    N = 200
    wakerad = np.linspace(1.,120.,N)
    a = np.linspace(1.,0.1,N)
    for i in range(N):
        wake = plt.Circle((x1+70.,H1), wakerad[i]/2., edgecolor='C0', linewidth=.3,fill=None, alpha=a[i])
        ax3.add_artist(wake)

    c1 = R1/35.

    #add blades
    hub1 = plt.Circle((x1,H1), 3*c1, color='black', fill=False, linewidth=1)
    ax3.add_artist(hub1)
    bladeX = np.array([3.,3.,5.,6.,10.,13.,18.,23.,28.,33.,38.,33.,28.,23.,18.,13.,7.,6.,3.])
    bladeY = -(np.array([2.,0.6,0.6,0.,0.,0.8,1.5,1.7,1.9,2.1,2.3,2.4,2.4,2.4,2.4,2.4,2.4,2.4,2.])-1.5)*1.5

    angle1 = 0.

    blade1X = bladeX*cos(radians(angle1))-bladeY*sin(radians(angle1))
    blade1Y = bladeX*sin(radians(angle1))+bladeY*cos(radians(angle1))

    blade2X = bladeX*cos(radians(angle1+180.))-bladeY*sin(radians(angle1+180.))
    blade2Y = bladeX*sin(radians(angle1+180.))+bladeY*cos(radians(angle1+180.))


    ax3.plot(blade1X*c1+x1, blade1Y*c1+H1, linewidth=1, color='black')
    ax3.plot(blade2X*c1+x1, blade2Y*c1+H1, linewidth=1, color='black')

    ax3.arrow(x1-30.,0.,0.,-30.,head_width=5.,head_length=5.,ec='C0',fc='C0')
    ax3.arrow(x1+30.,0.,0.,-30.,head_width=5.,head_length=5.,ec='C0',fc='C0')

    ax3.arrow(x1-40.,0.,0.,20.,head_width=5.,head_length=5.,ec='C1',fc='C1')
    ax3.arrow(x1+40.,0.,0.,-10.,head_width=5.,head_length=5.,ec='C1',fc='C1')

    tx = np.array([x1-3.,x1-3.,x1+3.,x1+3.])
    ty = np.array([H1-60.,H1-3.*c1,H1-3.*c1,H1-60.])
    ax3.plot(tx,ty,'-k')

    ax3.text(x1-r+5.,7.,'1',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax3.text(x1+r-5.,7.,'2',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')







    H1 = 0.
    D1 = 126.4
    R1 = D1/2.
    spacing = 0.8
    r = 126.4/2.
    x1 = 200-r
    N = 200
    wakerad = np.linspace(1.,120.,N)
    a = np.linspace(1.,0.1,N)
    for i in range(N):
        wake = plt.Circle((x1-70.,H1), wakerad[i]/2., edgecolor='C0', linewidth=.3,fill=None, alpha=a[i])
        ax5.add_artist(wake)

    ax5.text(40.,15.,'wake',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')

    c1 = R1/35.

    #add blades
    hub1 = plt.Circle((x1,H1), 3*c1, color='black', fill=False, linewidth=1)
    ax5.add_artist(hub1)
    bladeX = np.array([3.,3.,5.,6.,10.,13.,18.,23.,28.,33.,38.,33.,28.,23.,18.,13.,7.,6.,3.])
    bladeY = -(np.array([2.,0.6,0.6,0.,0.,0.8,1.5,1.7,1.9,2.1,2.3,2.4,2.4,2.4,2.4,2.4,2.4,2.4,2.])-1.5)*1.5

    angle1 = 0.

    blade1X = bladeX*cos(radians(angle1))-bladeY*sin(radians(angle1))
    blade1Y = bladeX*sin(radians(angle1))+bladeY*cos(radians(angle1))

    blade2X = bladeX*cos(radians(angle1+180.))-bladeY*sin(radians(angle1+180.))
    blade2Y = bladeX*sin(radians(angle1+180.))+bladeY*cos(radians(angle1+180.))


    ax5.plot(blade1X*c1+x1, blade1Y*c1+H1, linewidth=1, color='black')
    ax5.plot(blade2X*c1+x1, blade2Y*c1+H1, linewidth=1, color='black')

    ax5.arrow(x1-30.,0.,0.,-30.,head_width=5.,head_length=5.,ec='C0',fc='C0')
    ax5.arrow(x1+30.,0.,0.,-30.,head_width=5.,head_length=5.,ec='C0',fc='C0')

    ax5.arrow(x1-40.,0.,0.,10.,head_width=5.,head_length=5.,ec='C1',fc='C1')
    ax5.arrow(x1+40.,0.,0.,-20.,head_width=5.,head_length=5.,ec='C1',fc='C1')

    tx = np.array([x1-3.,x1-3.,x1+3.,x1+3.])
    ty = np.array([H1-60.,H1-3.*c1,H1-3.*c1,H1-60.])
    ax5.plot(tx,ty,'-k')

    ax5.text(x1-r+5.,7.,'1',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax5.text(x1+r-5.,7.,'2',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')



    ax1.axis('equal')
    ax3.axis('equal')
    ax5.axis('equal')

    ax1.set_xlim([0,200.])
    ax1.set_ylim([-120,120])
    ax3.set_xlim([0,200.])
    ax3.set_ylim([-120,120])
    ax5.set_xlim([0,200.])
    ax5.set_ylim([-120,120])





    def make_loads(maxL,minL):
        angles = np.linspace(0.,4.*np.pi,1000)
        amp = (maxL-minL)/2.
        loads = amp*np.sin(angles)
        return angles,loads

    xline = np.array([0.,4*np.pi])
    yline = np.zeros_like(xline)
    ax2.plot(xline,yline,'-k',linewidth=2)
    ax4.plot(xline,yline,'-k',linewidth=2)
    ax6.plot(xline,yline,'-k',linewidth=2)

    angles,loads = make_loads(1.,-0.5)
    ax2.plot(angles,loads+0.25+0.1)
    ax2.set_xlim(0.,4*np.pi)
    ax2.set_ylim(-1.5,1.5)


    angles,loads = make_loads(1.,-1)
    ax4.plot(angles,loads+0.1)
    ax4.set_xlim(0.,4*np.pi)
    ax4.set_ylim(-1.5,1.5)

    angles,loads = make_loads(0.5,-0.5)
    ax6.plot(angles,loads+0.1)
    ax6.set_xlim(0.,4*np.pi)
    ax6.set_ylim(-1.5,1.5)

    ax2.text(np.pi/2.,1.3,'2',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax2.text(5*np.pi/2.,1.3,'2',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax2.text(3*np.pi/2.,-0.7,'1',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax2.text(7*np.pi/2.,-0.7,'1',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')

    ax4.text(np.pi/2.,1.3,'2',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax4.text(5*np.pi/2.,1.3,'2',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax4.text(3*np.pi/2.,-1.2,'1',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax4.text(7*np.pi/2.,-1.2,'1',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')

    ax6.text(np.pi/2.,0.8,'2',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax6.text(5*np.pi/2.,0.8,'2',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax6.text(3*np.pi/2.,-0.7,'1',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')
    ax6.text(7*np.pi/2.,-0.7,'1',family='serif',fontsize=10,horizontalalignment='center',verticalalignment='center',color='black')




    ax2.plot(np.pi/2.,1.1,'ok',markersize=4.)
    ax2.plot(3*np.pi/2.,-0.4,'ok',markersize=4.)
    ax2.plot(5*np.pi/2.,1.1,'ok',markersize=4.)
    ax2.plot(7*np.pi/2.,-0.4,'ok',markersize=4.)

    ax4.plot(np.pi/2.,1.1,'ok',markersize=4.)
    ax4.plot(3*np.pi/2.,-0.9,'ok',markersize=4.)
    ax4.plot(5*np.pi/2.,1.1,'ok',markersize=4.)
    ax4.plot(7*np.pi/2.,-0.9,'ok',markersize=4.)

    ax6.plot(np.pi/2.,0.6,'ok',markersize=4.)
    ax6.plot(3*np.pi/2.,-0.4,'ok',markersize=4.)
    ax6.plot(5*np.pi/2.,0.6,'ok',markersize=4.)
    ax6.plot(7*np.pi/2.,-0.4,'ok',markersize=4.)

    ax2.set_yticks((-1.5,-1.,-0.5,0.,0.5,1.,1.5))
    ax2.set_yticklabels(('','','','','','',''))
    ax2.set_xticks((0.,np.pi,2*np.pi,3*np.pi,4*np.pi))
    ax2.set_xticklabels(('','','','',''))

    ax4.set_yticks((-1.5,-1.,-0.5,0.,0.5,1.,1.5))
    ax4.set_yticklabels(('','','','','','',''))
    ax4.set_xticks((0.,np.pi,2*np.pi,3*np.pi,4*np.pi))
    ax4.set_xticklabels(('','','','',''))

    ax6.set_yticks((-1.5,-1.,-0.5,0.,0.5,1.,1.5))
    ax6.set_yticklabels(('','','','','','',''))
    ax6.set_xticks((0.,np.pi,2*np.pi,3*np.pi,4*np.pi))
    ax6.set_xticklabels(('','','','',''))

    ax2.grid()
    ax4.grid()
    ax6.grid()

    ax2.set_ylabel('root loads',fontsize=10,color='black',family='serif',labelpad=-3.)
    ax4.set_ylabel('root loads',fontsize=10,color='black',family='serif',labelpad=-3.)
    ax6.set_ylabel('root loads',fontsize=10,color='black',family='serif',labelpad=-3.)

    ax6.set_xlabel('azimuth angle',fontsize=10,color='black',family='serif',labelpad=-3.)

    plt.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.02,
            hspace = 0.2, wspace = 0.1)

    # ax1.text(5,50,'unwaked case',family='serif',fontsize=10,horizontalalignment='left',verticalalignment='center',color='black')
    # ax3.text(5,50,'waked case 2',family='serif',fontsize=10,horizontalalignment='left',verticalalignment='center',color='black')
    # ax5.text(5,50,'waked case 1',family='serif',fontsize=10,horizontalalignment='left',verticalalignment='center',color='black')
    # plt.savefig('make_plots/figures/partial_loading.pdf',transparent=True)
    plt.show()
