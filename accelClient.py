from Accel import Accel #now you're the client
import matplotlib.pyplot as plt
import numpy as np

#initializes multiple at a time
def initialize(CIMT, TD, os, filetype):
    l = []
    for item in CIMT:
        l.append(Accel('CIMT' + item, applyButter = True, os = os, filetype = filetype))
    for item in TD:
        l.append(Accel('TD' + item, applyButter = True, os = os, filetype = filetype))
    return l

plt.close('all')
os = 'Mac'
filetype = 'Epoch'
TD08s = Accel('TD08', status = 'Sleep') #the s stands for sleep
TD08 = Accel('TD08')
TD08sum = [[] for i in range(4)]
for a, b in zip(TD08.UTM, TD08s.UTM):
    TD08sum[0].append(np.average(a))
    TD08sum[1].append(np.std(a))
    TD08sum[2].append(np.average(b))
    TD08sum[3].append(np.std(b))
TD08sum = np.array(TD08sum)
#l = initialize(['03', '04', '06', '08', '09'], ['01', '02', '03', '04', '05'], os, filetype) #for initializing multiple files
#for item in l:
#    item.ECDF(n = 50, kind = 'mag')
#CIMT03 = Accel('CIMT03', applyButter = True, os = os, filetype = filetype)
#CIMT04 = Accel('CIMT04', applyButter = True, os = os, filetype = filetype)
#CIMT06 = Accel('CIMT06', applyButter = True, os = os, filetype = filetype)
#CIMT08 = Accel('CIMT08', applyButter = True, os = os, filetype = filetype)
#CIMT09 = Accel('CIMT09', applyButter = True, os = os, filetype = filetype)