import stats
import unpack
import matplotlib.pyplot as plt

u = unpack.Unpack('PhotonCountDataE4/')
s = stats.Stats(u)

for fn in u.file_names:
    print fn

use = 4
print '----'
print u.file_names[use]
s.plot_histo(use, 20, gated=True, zoom_peak=False, log=True)
#s.plot_bin_of_bins(use)
"""
s.plot_histo(use, 500, gated=False, log=False)
s.plot_histo(use, 100, gated=False, zoom_peak=True, log=False)
s.plot_histo(use, 50, log=False)
s.plot_histo(use, 50)
s.plot_std_means()
s.plot_binned_events1(use)
s.plot_binned_events2(use)
s.plot_bin_of_bins(use)
"""