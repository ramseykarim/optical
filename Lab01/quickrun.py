import stats
import unpack

u = unpack.Unpack('PhotonCountDataE4/')
s = stats.Stats(u)
# for i in range(len(s.data.intervals)):
#     print s.data.file_names[i] + ': ', s.mean(i)

print s.data.file_names[0] + '\nMEAN: ', s.mean(0)

for i in range(len(s.data.intervals)):
    s.plot_mean_progressive(i, 100)
stats.show()
