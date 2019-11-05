
import numpy as np
import matplotlib.pyplot as plt
import gaus
import fast_calc_aep
import heapq


if __name__ == '__main__':

                nTurbs = 10.
                rotor_diameter = 126.4
                spacing = 3.
                side_length = (np.sqrt(nTurbs)-1.)*rotor_diameter*spacing
                a = side_length**2
                circle_radius = np.sqrt(a/np.pi)


                folder = '/Users/ningrsrch/Dropbox/Projects/waked-loads/yy_results/10turbs_2dirs_aep'

                fileAEP = folder+'/AEP.txt'
                fileX = folder+'/turbineX.txt'
                fileY = folder+'/turbineY.txt'
                fileDAMAGE = folder+'/damage.txt'

                damage = np.loadtxt(fileDAMAGE)
                maxD = np.zeros(100)
                for i in range(len(maxD)):
                        maxD[i] = np.max(damage[i])

                x = np.loadtxt(fileX)
                y = np.loadtxt(fileY)
                aep = np.loadtxt(fileAEP)
                maxD = maxD[0:100]

                """min or max"""
                l = np.argsort(maxD)
                ind = l[10]
                # ind = np.argmax(maxD)
                print aep[ind]
                # max 45.714167167805684
                # min 45.862487816411644


                D = damage[ind]
                mind = np.argmax(damage[ind])

                turbineX = x[ind,:]
                turbineY = y[ind,:]

                wind_speed = 8.

                wec_factor = 1.
                nTurbines = len(turbineX)
                turbineZ = np.ones(nTurbines)*90.
                yaw = np.zeros(nTurbines)
                rotorDiameter = np.ones(nTurbines)*126.4
                ky = 0.022
                kz = 0.022
                alpha = 2.32
                beta = 0.154

                # I = 0.056
                z_ref = 50.
                z_0 = 0.
                shear_exp = 0.
                RotorPointsY = np.array([0.])
                RotorPointsZ = np.array([0.])

                sorted_x_idx = np.argsort(turbineX)

                use_ct_curve = False
                interp_type = 1.
                Ct = np.ones(nTurbines)*8./9.
                ct_curve_wind_speed = np.ones_like(Ct)*wind_speed
                ct_curve_ct = Ct

                wake_model_version=2016
                sm_smoothing=700.
                calc_k_star=True
                ti_calculation_method=2
                wake_combination_method=1
                I=0.11

                res = 500
                x = np.linspace(-circle_radius-100.,circle_radius+100.,res)
                y = np.linspace(-circle_radius-100.,circle_radius+100.,res)
                velX, velY = np.meshgrid(x,y)
                velX = np.ndarray.flatten(velX)
                velY = np.ndarray.flatten(velY)
                velZ = np.ones_like(velX)*90.


                fig = plt.figure(1,figsize=[6.,2.5])
                ax1 = plt.subplot(121)
                ax2 = plt.subplot(122)

                from yy_calc_fatigue import *
                filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
                filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
                filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'

                TI = 0.11
                Omega_free,free_speed,Omega_close,close_speed,Omega_far,far_speed = find_omega(filename_free,filename_close,filename_far,TI=TI)
                Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()
                print Omega_free,free_speed,Omega_close,close_speed,Omega_far,far_speed

                dir = 270.
                turbineXw, turbineYw = fast_calc_aep.windframe(dir, turbineX, turbineY)

                ws_array, wake_diameters = gaus.porteagel_visualize(turbineXw, sorted_x_idx, turbineYw, turbineZ, rotorDiameter, Ct,
                           wind_speed, yaw, ky, kz, alpha, beta, I, RotorPointsY,
                           RotorPointsZ, z_ref, z_0, shear_exp, velX, velY, velZ,
                           wake_combination_method, ti_calculation_method, calc_k_star,
                           wec_factor, wake_model_version, interp_type, use_ct_curve,
                           ct_curve_wind_speed, ct_curve_ct, sm_smoothing)

                damage270 = farm_damage(turbineXw,turbineYw,np.array([270.]),np.array([0.5]),Omega_free,free_speed,
                                                    Omega_close,close_speed,Omega_far,far_speed,Rhub,r,chord,theta,af,Rtip,
                                                    B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI)

                print damage270

                x = np.zeros((res,res))
                y = np.zeros((res,res))
                z = np.zeros((res,res))
                for i in range(res):
                                for j in range(res):
                                                x[i][j] = velX[i*res+j]
                                                y[i][j] = velY[i*res+j]
                                                z[i][j] = ws_array[i*res+j]

                ax1.pcolormesh(x,y,z,vmin=0.,vmax=8.,cmap='Blues_r',shading='gourand')


                dir = 0.
                turbineXw, turbineYw = fast_calc_aep.windframe(dir, turbineX, turbineY)

                ws_array, wake_diameters = gaus.porteagel_visualize(turbineXw, sorted_x_idx, turbineYw, turbineZ, rotorDiameter, Ct,
                           wind_speed, yaw, ky, kz, alpha, beta, I, RotorPointsY,
                           RotorPointsZ, z_ref, z_0, shear_exp, velX, velY, velZ,
                           wake_combination_method, ti_calculation_method, calc_k_star,
                           wec_factor, wake_model_version, interp_type, use_ct_curve,
                           ct_curve_wind_speed, ct_curve_ct, sm_smoothing)



                damage0 = farm_damage(turbineXw,turbineYw,np.array([270.]),np.array([0.5]),Omega_free,free_speed,
                                                    Omega_close,close_speed,Omega_far,far_speed,Rhub,r,chord,theta,af,Rtip,
                                                    B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI)
                print damage0

                x = np.zeros((res,res))
                y = np.zeros((res,res))
                z = np.zeros((res,res))
                for i in range(res):
                                for j in range(res):
                                                x[i][j] = velX[i*res+j]
                                                y[i][j] = velY[i*res+j]
                                                z[i][j] = ws_array[i*res+j]

                ang = -np.pi/2.
                xt = np.cos(ang)*x-np.sin(ang)*y
                yt = np.sin(ang)*x+np.cos(ang)*y
                ax2.pcolormesh(xt,yt,z,vmin=0.,vmax=8.,cmap='Blues_r',shading='gourand')
                two = ax2.pcolormesh(xt,yt,z,vmin=0.,vmax=8.,cmap='Blues_r',shading='gourand')

                from mpl_toolkits.axes_grid1 import make_axes_locatable
                def colorbar(mappable):
                    ax = mappable.axes
                    fig = ax.figure
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("left", size="5%", pad=0.3)
                    return fig.colorbar(mappable, cax=cax)

                cbar_ax = fig.add_axes([0.9, 0.05, 0.03, 0.9])
                fig.colorbar(two, cax=cbar_ax)

                # plt.colorbar(z,ax=ax2)








                r = 126.4/2.
                for i in range(len(turbineX)):
                                if i==mind:
                                                color='red'
                                else:
                                                color='black'
                                x = np.array([turbineX[i],turbineX[i]])
                                y = np.array([turbineY[i]-r,turbineY[i]+r])
                                ax1.plot(x,y,'-',color=color,linewidth=3)
                                x = np.array([turbineX[i]-r,turbineX[i]+r])
                                y = np.array([turbineY[i],turbineY[i]])
                                ax2.plot(x,y,'-',color=color,linewidth=3)

                                ax1.text(turbineX[i],turbineY[i]+100,'%s'%round(damage270[i],2),color=color,fontsize=8,family='serif',horizontalalignment='center',verticalalignment='center',weight='bold')
                                ax2.text(turbineX[i],turbineY[i]+30,'%s'%round(damage0[i],2),color=color,fontsize=8,family='serif',horizontalalignment='center',verticalalignment='center',weight='bold')

                bound = plt.Circle((0.,0.),circle_radius, color='black', fill=False, linewidth=1)
                ax1.add_artist(bound)
                bound = plt.Circle((0.,0.),circle_radius, color='black', fill=False, linewidth=1)
                ax2.add_artist(bound)

                ax1.axis('equal')
                ax1.axis('off')
                ax2.axis('equal')
                ax2.axis('off')


                plt.subplots_adjust(top = 0.98, bottom = 0.02, right = 0.9, left = 0.02,
                hspace = 0, wspace = 0.05)

                ax2.text(830.,0.,'m/s',rotation=-90.,verticalalignment='center',horizontalalignment='center',fontsize=10,family='serif',style='italic')

                # plt.savefig('make_plots/figures/wakes_highdamage.png',transparent=True)
                plt.show()

                # # fig = plt.figure(1,figsize=[6.,6.])
                # # ax1 = plt.subplot(111)
                # #
                # #
                # # import matplotlib as mpl
                # # label_size = 8
                # # mpl.rcParams['xtick.labelsize'] = label_size
                # # mpl.rcParams['ytick.labelsize'] = label_size
                # #
                # # fontProperties = {'family':'serif','size':8}
                # # ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
                # # ax1.set_yticklabels(ax1.get_yticks(), fontProperties)
                # #
                # #

                # #
                # #
                # # #
                # # plt.show()
