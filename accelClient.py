from Accel import Accel #now you're the client
import matplotlib.pyplot as plt

plt.close('all')
CIMT03 = Accel('CIMT03', applyButter = True)
CIMT03.ECDF()
