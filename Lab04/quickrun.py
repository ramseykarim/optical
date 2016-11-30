import calibration as cal
import matplotlib.pyplot as plt
import unpacking as up
import solar


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
s.generate_light_noise()
plt.show()
