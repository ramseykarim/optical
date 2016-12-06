import numpy as np
import calibration as cal
import matplotlib.pyplot as plt
import unpacking as up
import solar
import sys


# u = up.Unpack()
# plt.figure()
# u.plot_laser()
# plt.figure()
# u.plot_neon()
# plt.figure()
# u.plot_halogen()
#

# calibrator = cal.Calibration()

# calibrator.examine_flat()

# calibrator.plot_integrate_laser()
# calibrator.plot_integrate_neon()

s = solar.Sun()
s.get_slope()
plt.show()
sys.exit(0)
wc, sc = s.calibrate_sun(s.suns[s.center])
first = np.where(s.curve > np.mean(s.curve) * 2.)[0][-1]
wf, sf = s.calibrate_sun(s.suns[first])
plt.plot(wc, sc)
plt.plot(wf, sf)
plt.show()

