from Accel import Accel #now you're the client
import matplotlib.pyplot as plt
import numpy as np

#initializes multiple at a time
def initialize(CIMT, TD, os, filetype):
    l = []
    for item in CIMT:
        l.append(Accel('CIMT' + item, os, filetype, applyButter = True))
    for item in TD:
        l.append(Accel('TD' + item, os, filetype, applyButter = True))
    return l

def asleepVSawake(wake, sleep):
    from scipy import stats as st
    stats = [[] for i in range(4)]
    for a, b in zip(wake.UTM, sleep.UTM):
        stats[0].append(np.average(a))
        stats[1].append(np.std(a))
        stats[2].append(np.average(b))
        stats[3].append(np.std(b))
    stats = np.array(stats)
    for i in range(6):
        plt.figure()
        plt.title(wake.titles[i])
        mu1, std1, mu2, std2 = stats[:, i]
        x1 = np.linspace(mu1 - 3 * std1, mu1 + 3 * std1, 100)
        x2 = np.linspace(mu2 - 3 * std2, mu2 + 3 * std2, 100)
        plt.plot(x1, st.norm.pdf(x1, mu1, std1), alpha = 0.5)
        plt.plot(x2, st.norm.pdf(x2, mu2, std2), alpha = 0.5)
        plt.xlabel('magnitude')
        plt.ylabel('probability density')
        plt.legend(['awake', 'asleep'])
    return stats
    

plt.close('all')

TD08s = Accel('TD08', 'Mac', 'Epoch', status = 'Sleep') #the s stands for sleep
TD08 = Accel('TD08', 'Mac', 'Epoch')
#TD08sum = asleepVSawake(TD08, TD08s)
    
#l = initialize(['03', '04', '06', '08', '09'], ['01', '02', '03', '04', '05'], os, filetype) #for initializing multiple files
#for item in l:
#    item.ECDF(n = 50, kind = 'mag')
