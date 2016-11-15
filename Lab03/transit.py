import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst

T_DUR = 2.5  # Hours
P_ORB = 2.79  # Days
RR = np.arange(0.007, 0.030, 0.005)
B_ARRAY = np.arange(0.5, 1.05, 0.05)
R_STAR = 1.428  # Solar Radius
SR_AU = 695.7e6 / cst.au


def radius(b, rr):
    first = (1. + np.sqrt(rr)) ** 2.
    inside_root = first - (b ** 2.)
    inside_sin = np.pi * T_DUR / (P_ORB * 24.)
    return (np.sqrt(inside_root) / np.sin(inside_sin)) * R_STAR * SR_AU


plt.figure()
leg = []
for r in RR[::-1]:
    plt.plot(B_ARRAY, radius(B_ARRAY, r), '-')
    leg.append("$b = ${0:.3f}".format(r))
plt.xlim([0.45, 1.05])
plt.ylim([0.02, 0.06])
plt.legend(leg, loc='upper right')
plt.xlabel("Impact parameter $b$")
plt.ylabel("Semimajor axis $a$ (AU)")
plt.title("Estimate of Orbital Parameter $a$ of HAT-P-56 b")
plt.show()
