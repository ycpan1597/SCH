from Accel import Accel #now you're the client
import matplotlib.pyplot as plt
import numpy as np

#initializes multiple at a time
def initialize(subjects, OS, filetype):
    l = {}
    for item in subjects:
        l[str(item)] = Accel(item, OS, filetype, applyButter = True)
    return l


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

#dic = initialize(['TD01', 'TD02', 'TD03', 'TD04', 'TD05', 'TD06', 'TD07', 'TD08',
#                  'CIMT03', 'CIMT04', 'CIMT06', 'CIMT08', 'CIMT09', 'CIMT13'], 'Baker', 'Raw')
#for key, value in dic.items():
#    value.michaelsRatio()

TD, CIMT = [], []
for key, value in dic.items():
    if 'TD' in key and value.age in range(6, 10):
        TD.append(value.michaelsRatio())
    elif 'CIMT' in key and value.age in range(6, 10):
        CIMT.append(value.michaelsRatio())

TDavg = np.mean(TD)
TDvar = np.std(TD)
pre, during, post = np.mean(CIMT, axis = 0)

plt.figure()
plt.axvline(x = 0.5, color = 'y', label = 'Bimanual')
plt.axvline(x = TDavg, color = 'b', label = 'TD')
plt.axvline(x = TDavg + TDvar, color = 'b', ls = '--', label = 'TD range')
plt.axvline(x = TDavg - TDvar, color = 'b', ls = '--')
plt.axvline(x = pre, color = 'g', ls = '-.', label = 'CIMT (pre)')
plt.axvline(x = during, color = 'k', ls = '-.', label = 'CIMT (during)')
plt.axvline(x = post, color = 'r', ls = '-.', label = 'CIMT (post)')
plt.axis([0.4, 0.6, 0, 3])
plt.text(0.4, 2.8, 'pre: %.3f' % pre, color = 'g')
plt.text(0.4, 2.6, 'during: %.3f' % during, color = 'k')
plt.text(0.4, 2.4, 'post: %.3f' % post, color = 'r')
plt.legend()
plt.title('Average Median of Jerk Ratio')
plt.xlabel('Jerk Ratio')