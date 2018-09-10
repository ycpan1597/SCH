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
    
#%%     
TD08sum = np.array(TD08sum)
import matplotlib.mlab as mlab
for i in range(6):
    plt.figure()
    plt.title(TD08.titles[i])
    mu1, std1, mu2, std2 = TD08sum[:, i]
    x1 = np.linspace(mu1 - 3 * std1, mu1 + 3 * std1, 100)
    x2 = np.linspace(mu2 - 3 * std2, mu2 + 3 * std2, 100)
    plt.plot(x1, mlab.normpdf(x1, mu1, std1), alpha = 0.5)
    plt.plot(x2, mlab.normpdf(x2, mu2, std2), alpha = 0.5)
    plt.xlabel('magnitude')
    plt.ylabel('probability density')
    plt.legend(['awake', 'asleep'])
#l = initialize(['03', '04', '06', '08', '09'], ['01', '02', '03', '04', '05'], os, filetype) #for initializing multiple files
#for item in l:
#    item.ECDF(n = 50, kind = 'mag')
#CIMT03 = Accel('CIMT03', applyButter = True, os = os, filetype = filetype)
#CIMT04 = Accel('CIMT04', applyButter = True, os = os, filetype = filetype)
#CIMT06 = Accel('CIMT06', applyButter = True, os = os, filetype = filetype)
#CIMT08 = Accel('CIMT08', applyButter = True, os = os, filetype = filetype)
#CIMT09 = Accel('CIMT09', applyButter = True, os = os, filetype = filetype)