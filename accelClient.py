from Accel import Accel #now you're the client
import matplotlib.pyplot as plt

plt.close('all')
CIMT03 = Accel('CIMT03', applyButter = True, os = 'Baker', filetype = 'Raw')
#CIMT03.ECDF()
TD05 = Accel('TD05', applyButter = True, os = 'Baker', filetype = 'Raw')