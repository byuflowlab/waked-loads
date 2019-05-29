import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""colors"""
robin = '#B2DBD5'
nectar = '#EAB364'
tuscan = '#B2473E'
olive = '#ACBD78'

AEP = np.array([5.59472761, 5.57886797, 5.57003505, 5.57134922, 5.55492697,
       5.55240162, 5.53442257, 5.5342501 , 5.50908267, 5.49701015,
       5.47796257, 5.44173408, 5.41255775, 5.37265848, 5.3233819 ,
       5.32457509, 5.2784313 , 5.26338343])*100.
overlap = np.array([14.65595887, 14.50001963, 14.00015199, 13.50003129, 13.00007449,
       12.50013801, 12.00013877, 11.50003943, 11.00000144, 10.50000355,
       10.00011554,  9.50007155,  9.00004438,  8.50003524,  8.00025851,
        7.500001  ,  7.00000038,  6.50007247])/20. * 100.

fig = plt.figure(figsize=[3.25,2.25])
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=9)
ax.tick_params(axis='both', which='minor', labelsize=9)

ax.plot(overlap,AEP,'o',color=nectar)


ax.set_ylabel('AEP (GWh)', fontsize=9)
ax.set_xlabel('average % waked', fontsize=9)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')


plt.tight_layout()
plt.savefig('preliminary.pdf',transparent=True)
plt.show()
