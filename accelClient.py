from Accel import Accel #now you're the client
import matplotlib.pyplot as plt
import numpy as np

#initializes multiple at a time
def initialize(subjects, OS, filetype):
    l = {}
    for item in subjects:
        l[str(item)] = Accel(item, OS, filetype, applyButter = True)
    return l

#def initialize(CIMT, TD, OS, filetype):
#    l = []
#    for item in CIMT:
#        l.append(Accel('CIMT' + item, OS, filetype, applyButter = True))
#    for item in TD:
#        l.append(Accel('TD' + item, OS, filetype, applyButter = True))
#    return l

def asleepVSawake(awake, asleep):
    from scipy import stats as st
    stats = [[] for i in range(4)]
    for a, b in zip(awake.UTM, asleep.UTM):
        stats[0].append(np.average(a))
        stats[1].append(np.std(a))
        stats[2].append(np.average(b))
        stats[3].append(np.std(b))
    stats = np.array(stats)
    for i in range(6):
        plt.figure()
        plt.title(awake.titles[i])
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

dic = initialize(['TD01', 'TD02', 'TD03', 'TD04', 'TD05', 'TD06', 'TD07', 'TD08',
                  'CIMT03', 'CIMT06', 'CIMT08', 'CIMT09'], 'Mac', 'Epoch')
for key, value in dic.items():
    value.compareJerk()
