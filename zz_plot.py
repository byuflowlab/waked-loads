
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':


            offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])

            #VONMISES
            # FAST4 = np.array([0.60992012, 0.65165644, 0.75169803, 0.76573425, 0.69897811,
            #             0.80055786, 0.88997082, 0.81021015, 0.6787323 ])
            # FAST7 = np.array([0.59931602, 0.63837788, 0.64219613, 0.68481489, 0.68044648,
            #             0.72837558, 0.74167147, 0.7535786 , 0.68715821])
            # FAST10 = np.array([0.57622497, 0.61632883, 0.61795598, 0.61048667, 0.6514622 ,
            #             0.66286093, 0.68190521, 0.67172391, 0.66623172])
            #
            # super4 = np.array([0.51705482, 0.63874493, 0.71331363, 0.78463531, 0.8020471 ,
            #             0.78098615, 0.77429964, 0.65715762, 0.55646555])
            # super7 = np.array([0.50735255, 0.59099391, 0.64568728, 0.63682097, 0.61003009,
            #             0.62945768, 0.63844614, 0.52249784, 0.46192422])
            # super10 = np.array([0.58062445, 0.68850039, 0.6859951 , 0.66806091, 0.68617964,
            #             0.67947511, 0.66462743, 0.67355746, 0.59818434])

            #FLAPWISE
            FAST4 = np.array([0.77208607, 0.84301158, 0.98214149, 1.0404579 , 1.01968396,
                        1.05293275, 1.02057731, 0.9109089 , 0.7991055 ])
            FAST7 = np.array([0.79800391, 0.85273074, 0.90235949, 0.9509322 , 1.00282632,
                        0.95212582, 0.92136755, 0.89915201, 0.84002526])
            FAST10 = np.array([0.79923674, 0.88938653, 0.87021251, 0.86827795, 0.89101762,
                        0.91150866, 0.90512195, 0.85187077, 0.82026554])

            #1.75
            # super4 = np.array([0.64737379, 0.71410274, 0.84372975, 1.01488423, 1.02441436,
            #             1.02304399, 0.85607482, 0.75953559, 0.66188621])
            # super7 = np.array([0.70232316, 0.84054196, 0.93208034, 0.99455852, 0.95606191,
            #             0.97013588, 0.98067069, 0.88536442, 0.71664505])
            # super10 = np.array([0.88267287, 0.91994913, 0.91786632, 0.93059056, 0.89101762,
            #             0.92375509, 0.91782382, 0.93243011, 0.86312683])

            #2.0
            super4 = np.array([0.66494885, 0.81146866, 0.91370116, 1.0153531 , 1.02441436,
                        1.0278231 , 0.91861657, 0.80785384, 0.70177373])
            super7 = np.array([0.79889875, 0.90225785, 0.93659093, 0.99455852, 0.95606191,
                        0.97013588, 0.98402243, 0.95863066, 0.79036316])
            super10 = np.array([0.9213668 , 0.92088275, 0.91786632, 0.93059056, 0.89101762,
                        0.92375509, 0.91782382, 0.93273836, 0.90722254])

            #2.6
            # super4 = np.array([0.86388579, 0.96110712, 0.95467141, 1.0153531 , 1.02441436,
            #             1.0278231 , 0.97084534, 0.9645925 , 0.89015409])
            # super7 = np.array([0.95759255, 0.92761633, 0.93659093, 0.99455852, 0.95606191,
            #             0.97013588, 0.98402243, 0.97808599, 0.96069176])
            # super10 = np.array([0.93006787, 0.92088275, 0.91786632, 0.93059056, 0.89101762,
            #             0.92375509, 0.91782382, 0.93273836, 0.92181197])

            #3.0
            # super4 = np.array([0.95750355, 0.99252945, 0.95467141, 1.0153531 , 1.02441436,
            #             1.0278231 , 0.97084534, 0.99635835, 0.98314488])
            # super7 = np.array([0.96169386, 0.92761633, 0.93659093, 0.99455852, 0.95606191,
            #             0.97013588, 0.98402243, 0.97808599, 0.96835489])
            # super10 = np.array([0.93006787, 0.92088275, 0.91786632, 0.93059056, 0.89101762,
            #             0.92375509, 0.91782382, 0.93273836, 0.92181197])

       #      super4 = np.array([0.86704918, 0.95836709, 0.95351321, 1.0157565 , 1.03939396,
       # 1.02664699, 0.97165432, 0.9657923 , 0.89213373])
       #      super7 = np.array([0.95996   , 0.92998632, 0.93698802, 0.99692455, 0.96552271,
       # 0.96895387, 0.9867843 , 0.97611672, 0.96187821])
       #      super10 = np.array([0.92809633, 0.9208839 , 0.91707893, 0.93098476, 0.91269862,
       # 0.9241498 , 0.9174291 , 0.93313335, 0.92062992])





            dam4 = np.array([0.72483608, 0.83937936, 0.96142018, 1.01968396, 1.01968396,
                        1.01968396, 0.96142018, 0.83937936, 0.72483608])
            dam7 = np.array([0.8210742 , 0.91986158, 0.95535079, 0.95535079, 0.95535079,
                        0.95535079, 0.95535079, 0.91986158, 0.8210742 ])
            dam10 = np.array([0.87051678, 0.89101762, 0.89101762, 0.89101762, 0.89101762,
                        0.89101762, 0.89101762, 0.89101762, 0.87051678])

            fig = plt.figure(1,figsize=[6.,2.5])
            ax1 = plt.subplot(131)
            ax2 = plt.subplot(132)
            ax3 = plt.subplot(133)

            import matplotlib as mpl
            label_size = 8
            mpl.rcParams['xtick.labelsize'] = label_size
            mpl.rcParams['ytick.labelsize'] = label_size

            fontProperties = {'family':'serif','size':8}
            ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
            ax1.set_yticklabels(ax1.get_yticks(), fontProperties)

            ax2.set_xticklabels(ax2.get_xticks(), fontProperties)
            ax2.set_yticklabels(ax2.get_yticks(), fontProperties)

            ax3.set_xticklabels(ax3.get_xticks(), fontProperties)
            ax3.set_yticklabels(ax3.get_yticks(), fontProperties)

            offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])

            ax1.plot(offset,FAST4,'o',label='SOWFA+FAST',color='C1',markersize=4)
            ax1.plot(offset,super4,'o',label='superimposed loads',color='C0',markersize=4)
            ax1.plot(offset,dam4,'o',label='superimposed damage',color='C2',markersize=4)

            # ax1.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black',label='freestream loads')
            ax1.legend(loc=3,prop={'family':'serif', 'size':8})

            ax1.set_title('7D',family='serif',fontsize=10)
            ax1.set_ylabel('damage',family='serif',fontsize=10)

            ax1.set_xlabel('offset (D)',family='serif',fontsize=10)


            ax2.plot(offset,FAST7,'or',color='C1',markersize=4)
            ax2.plot(offset,super7,'ob',color='C0',markersize=4)
            ax2.plot(offset,dam7,'o',color='C2',markersize=4)
            # ax2.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
            ax2.set_title('7D',family='serif',fontsize=10)
            ax2.set_xlabel('offset (D)',family='serif',fontsize=10)
            #

            ax3.plot(offset,FAST10,'or',color='C1',markersize=4)
            ax3.plot(offset,super10,'ob',color='C0',markersize=4)
            ax3.plot(offset,dam10,'o',color='C2',markersize=4)
            ax3.set_title('10D',family='serif',fontsize=10)
            # ax3.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
            ax3.set_xlabel('offset (D)',family='serif',fontsize=10)


            ax1.set_yticks((0.25,0.5,0.75,1.0,1.1))
            ax1.set_yticklabels(('0.25','0.5','0.75','1.0',''))
            ax2.set_yticks((0.25,0.5,0.75,1.0,1.1))
            ax2.set_yticklabels(('','','','',''))
            ax3.set_yticks((0.25,0.5,0.75,1.0,1.1))
            ax3.set_yticklabels(('','','','',''))


            ax1.set_xlim(-1.1,1.1)
            ax2.set_xlim(-1.1,1.1)
            ax3.set_xlim(-1.1,1.1)

            ax1.set_xticks((-1.,-0.5,0.,0.5,1.))
            ax1.set_xticklabels(('-1','-0.5','0','0.5','1'))
            ax2.set_xticks((-1.,-0.5,0.,0.5,1.))
            ax2.set_xticklabels(('-1','-0.5','0','0.5','1'))
            ax3.set_xticks((-1.,-0.5,0.,0.5,1.))
            ax3.set_xticklabels(('-1','-0.5','0','0.5','1'))

            ax1.set_title('4D downstream',family='serif',fontsize=10)
            ax2.set_title('7D downstream',family='serif',fontsize=10)
            ax3.set_title('10D downstream',family='serif',fontsize=10)

            plt.subplots_adjust(top = 0.8, bottom = 0.2, right = 0.98, left = 0.1,
            hspace = 0, wspace = 0.1)

            ax1.set_ylim(0.25,1.1)
            ax2.set_ylim(0.25,1.1)
            ax3.set_ylim(0.25,1.1)

            plt.suptitle('11% TI',family='serif',fontsize=10)
            # plt.savefig('make_plots/figures/fatigue_damage11.pdf',transparent=True)
            plt.show()
