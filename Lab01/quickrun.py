import stats
import unpack

u = unpack.Unpack('PhotonCountDataE4/')
s = stats.Stats(u)

#print u.file_names

use = 5

print u.file_names[use]

s.plot_std_means()