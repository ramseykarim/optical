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

c = cal.Calibration()

# c.examine_flat()

# c.plot_integrate_laser()
c.plot_integrate_neon()


# s = solar.Sun()
# s.light_curve()

plt.show()
