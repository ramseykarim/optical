import stats
import unpack

u = unpack.Unpack('PhotonCountDataE4/')
s = stats.Stats(u)

s.plot_sdom_by_rootn(0)

