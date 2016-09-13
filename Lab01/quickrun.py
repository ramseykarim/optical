import stats
import unpack

u = unpack.Unpack('PhotonCountDataE4/')
s = stats.Stats(u)

#print u.file_names

use = 5

print u.file_names[use]

s.plot_histo(use, 500, gated=False, log=False)
s.plot_histo(use, 100, gated=False, zoom_peak=True, log=False)
s.plot_histo(use, 50, log=False)
s.plot_histo(use, 50)
s.plot_std_means()
s.plot_binned_events1(use)
s.plot_binned_events2(use)
s.plot_bin_of_bins(use)
