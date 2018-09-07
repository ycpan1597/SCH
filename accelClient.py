from Accel import Accel #now you're the client
import matplotlib.pyplot as plt

def initialize(CIMT, TD, os, filetype):
    l = []
    for item in CIMT:
        l.append(Accel('CIMT' + item, applyButter = True, os = os, filetype = filetype))
    for item in TD:
        l.append(Accel('TD' + item, applyButter = True, os = os, filetype = filetype))
    return l

plt.close('all')
<<<<<<< HEAD
os = 'Baker'
filetype = 'Raw'
l = initialize(['03', '04', '06'], ['01', '02', '03'], os, filetype)
=======
os = 'Mac'
filetype = 'Epoch'
l = initialize(['03', '04', '06', '08', '09'], ['01', '02', '03', '04', '05'], os, filetype) #for initializing multiple files
>>>>>>> 75e887d7ee4c6f7035dc99cd99706ba828c8d999
#CIMT03 = Accel('CIMT03', applyButter = True, os = os, filetype = filetype)
#CIMT04 = Accel('CIMT04', applyButter = True, os = os, filetype = filetype)
#CIMT06 = Accel('CIMT06', applyButter = True, os = os, filetype = filetype)
#CIMT08 = Accel('CIMT08', applyButter = True, os = os, filetype = filetype)
#CIMT09 = Accel('CIMT09', applyButter = True, os = os, filetype = filetype)
#
#TD05 = Accel('TD05', applyButter = True, os = os, filetype = filetype)
