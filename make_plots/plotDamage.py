
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

        offset = np.array([-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.])

        FAST4 = np.array([0.7479882 , 0.79418754, 0.91482836, 0.95437402, 0.9302589 ,0.96215701, 0.95952926, 0.86900191, 0.77281552])
        FAST7 = np.array([0.77931478, 0.82502943, 0.86154222, 0.89375726, 0.94159877, 0.8970284 , 0.87188754, 0.86078396, 0.81570256])
        FAST10 = np.array([0.7664489 , 0.85394419, 0.82298162, 0.83205289, 0.84391592, 0.85639593, 0.86367084, 0.82085985, 0.78787999])

        FAST4 = np.array([0.77208607, 0.84301158, 0.98214149, 1.0404579 , 1.01968396,
             1.05293275, 1.02057731, 0.9109089 , 0.7991055 ])
        FAST7 = np.array([0.79800391, 0.85273074, 0.90235949, 0.9509322 , 1.00282632,
       0.95212582, 0.92136755, 0.89915201, 0.84002526])
        FAST10 = np.array([0.79923674, 0.88938653, 0.87021251, 0.86827795, 0.89101762,
       0.91150866, 0.90512195, 0.85187077, 0.82026554])

        FASTfree = np.ones_like(FAST4)*0.6373399523538535



      #  #2.0
      #   super4 = np.array([0.71531298, 0.83092194, 0.95409818, 1.01290402, 1.01290402,
      #  1.01290402, 0.95409818, 0.83092194, 0.71531298])
      #   super7 = np.array([0.81251641, 0.91225896, 0.94809132, 0.94809132, 0.94809132,
      # 0.94809132, 0.94809132, 0.91225896, 0.81251641])
      #   super10 = np.array([0.86256839, 0.88327863, 0.88327863, 0.88327863, 0.88327863,
      #  0.88327863, 0.88327863, 0.88327863, 0.86256839])
      #
      #  #1.0
      #   super4 = np.array([0.61927481, 0.648916  , 0.73111831, 0.83236387, 0.89202716,
      #  0.83236387, 0.73111831, 0.648916  , 0.61927481])
      #   super7 = np.array([0.62429425, 0.68400553, 0.77196834, 0.87352005, 0.94809132,
      #  0.87352005, 0.77196834, 0.68400553, 0.62429425])
      #   super10 = np.array([0.64710586, 0.71277139, 0.79340773, 0.87266479, 0.88327863,
      #  0.87266479, 0.79340773, 0.71277139, 0.64710586])
      #
      #   #1.75
      #   super4 = np.array([0.67356624, 0.77695019, 0.89938595, 1.00926579, 1.01290402,
      #  1.00926579, 0.89938595, 0.77695019, 0.67356624])
      #   super7 = np.array([0.75322667, 0.85652751, 0.94390874, 0.94809132, 0.94809132,
      #  0.94809132, 0.94390874, 0.85652751, 0.75322667])
      #   super10 = np.array([0.80977114, 0.87877124, 0.88327863, 0.88327863, 0.88327863,
      #  0.88327863, 0.88327863, 0.87877124, 0.80977114])
      #

       #  super4 = np.array([0.62851456, 0.72823143, 0.84632434, 0.95230667, 0.95581586,
       # 0.95230667, 0.84632434, 0.72823143, 0.62851456])
       #  super7 = np.array([0.70720957, 0.80828081, 0.89377603, 0.89786834, 0.89786834,
       # 0.89786834, 0.89377603, 0.80828081, 0.70720957])
       #  super10 = np.array([0.7664779 , 0.83541739, 0.83992083, 0.83992083, 0.83992083,
       # 0.83992083, 0.83992083, 0.83541739, 0.7664779 ])

       #w/ Goodman
       #  super4 = np.array([0.68347414, 0.78590512, 0.90721229, 1.01607926, 1.01968396,
       # 1.01607926, 0.90721229, 0.78590512, 0.68347414])
       #  super7 = np.array([0.76235224, 0.86466383, 0.95120826, 0.95535079, 0.95535079,
       # 0.95535079, 0.95120826, 0.86466383, 0.76235224])
       #  super10 = np.array([0.81825337, 0.8865558 , 0.89101762, 0.89101762, 0.89101762,
       # 0.89101762, 0.89101762, 0.8865558 , 0.81825337])

        super4 = np.array([0.68347414, 0.78590512, 0.90721229, 1.01607926, 1.01968396,
       1.01607926, 0.90721229, 0.78590512, 0.68347414])
        super7 = np.array([0.76235224, 0.86466383, 0.95120826, 0.95535079, 0.95535079,
       0.95535079, 0.95120826, 0.86466383, 0.76235224])
        super10 = np.array([0.81825337, 0.8865558 , 0.89101762, 0.89101762, 0.89101762,
       0.89101762, 0.89101762, 0.8865558 , 0.81825337])

      #
      #






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
        ax1.plot(offset,FAST4,'o',label='SOWFA+FAST',color='C1')
        ax1.plot(offset,super4,'o',label='superimposed',color='C0')
        ax1.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black',label='freestream loads')
        # ax1.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0',label='freestream superimposed')
        ax1.legend(loc=3,prop={'family':'serif', 'size':8})

        ax1.set_title('7D',family='serif',fontsize=10)
        ax1.set_ylabel('damage',family='serif',fontsize=10)

        ax1.set_xlabel('offset (D)',family='serif',fontsize=10)

        ax2.plot(offset,FAST7,'or',color='C1')
        ax2.plot(offset,super7,'ob',color='C0')
        # ax2.plot(offset,FASTfree,'ok')
        # ax2.plot(offset,freestreamCC,'og')
        ax2.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
        # ax2.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0')
        ax2.set_title('7D',family='serif',fontsize=10)
        ax2.set_xlabel('offset (D)',family='serif',fontsize=10)
        #
        ax3.plot(offset,FAST10,'or',color='C1')
        ax3.plot(offset,super10,'ob',color='C0')
        # ax3.plot(offset,FASTfree,'ok')
        # ax3.plot(offset,freestreamCC,'og')
        ax3.set_title('10D',family='serif',fontsize=10)
        ax3.plot(np.array([-2.,2.]),np.array([1.,1.])*FASTfree[0],'--',color='black')
        # ax3.plot(np.array([-2.,2.]),np.array([1.,1.])*freestreamCC[0],'--',color='C0')
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

        plt.subplots_adjust(top = 0.9, bottom = 0.2, right = 0.98, left = 0.1,
                    hspace = 0, wspace = 0.1)

        ax1.set_ylim(0.25,1.1)
        ax2.set_ylim(0.25,1.1)
        ax3.set_ylim(0.25,1.1)

        # plt.suptitle('1.5')
        # plt.savefig('figures/fatigue_damage.pdf',transparent=True)
        plt.show()
