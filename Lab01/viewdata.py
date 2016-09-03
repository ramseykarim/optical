import numpy as np
import matplotlib.pyplot as plt
import os

data = []
directory = 'PhotonCountData/'
filenames = []
for filename in os.listdir(directory):

    data_raw = np.loadtxt(directory + filename,
                          delimiter=',', dtype='int32')
    ticktimes = data_raw[:,1]
    print ticktimes.shape
    data.append(ticktimes)
    filenames.append(filename)

data = np.array(data)

print data.shape
dimx = data[:,0].size
dimy = data[0,:].size
print dimx
print dimy
for i in range(dimx):
    plt.figure(i)
    plt.hist(data[i,:])
    plt.title(filenames[i])

#plt.plot(range(ticktimes.size), ticktimes)
plt.show()
