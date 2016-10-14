import unpack as up
import calibration as cal
import numpy as np
import matplotlib.pyplot as plt
import wavelength_lls as lls
import sky_spectra as sky
import plot_pipeline as pltt
import sys

u = up.Unpack()
c = cal.Calibrate(u)
pixels = np.arange(2048)


deg2_bs = c.wavelength_calibration
peak_wls_bs = np.array([deg2_bs[int(x)] for x in c.peaks])
deg2comp_bs_1 = lls.ALL_PEAK_ARRAY - peak_wls_bs
print "Residuals BS 1: ", np.sum(deg2comp_bs_1 ** 2.)

# s = sky.SkySpectra(u, c)
# s.plot_spectra()
# power, coefficient = s.fit_power_law()
# s.plot_power_law_fit(power, coefficient)

daytime_raw = up.average_run(u.obtain('sky'))
daytime_dark = up.average_run(u.obtain('darks_100'))
daytime_dark_sub = up.average_run(cal.dark_adjust(u.obtain('sky'), daytime_dark))

neon_raw = up.average_run(u.obtain('neon'))
neon_dark = up.average_run(u.obtain('darks_100'))
neon_dark_sub = up.average_run(cal.dark_adjust(u.obtain('neon'), neon_dark))

plt.figure(1)
plt.errorbar(c.wavelength_calibration, daytime_raw,
             xerr=c.wavelength_error(pixels), fmt=',', color='red')
plt.errorbar(c.wavelength_calibration, daytime_dark_sub,
             xerr=c.wavelength_error(pixels), fmt=',', color='black')
plt.errorbar(c.wavelength_calibration, neon_raw,
             xerr=c.wavelength_error(pixels), fmt=',', color='red')
plt.errorbar(c.wavelength_calibration, neon_dark_sub,
             xerr=c.wavelength_error(pixels), fmt=',', color='black')
plt.legend(['Raw', 'Dark Subtracted'], loc='upper left')
plt.xlabel(r"Wavelength ($\AA$)", fontsize=16, family='serif')
plt.ylabel("ADU", fontsize=16, family='serif')
plt.title("Raw ADU versus Dark Subtracted", fontsize=20, family='serif')


# plt.figure(1)
# cal.plot_peaks(c.averaged_cal_spectrum, c.peaks)
# plt.xlabel("Pixel", fontsize=16, family='serif')
# plt.ylabel("ADU", fontsize=16, family='serif')
# plt.title("Measured Spectra for Ne and Hg with peaks", fontsize=20, family='serif')
#
# plt.figure(2)
# plt.subplot(211)
# plt.plot(lls.ALL_PEAK_ARRAY, c.peaks, 'o', color='blue')
# plt.plot(deg2_bs, pixels, '-', color='k')
# plt.ylabel("Pixel", fontsize=16, family='serif')
# plt.title("Wavelength Solution to Spectrometer - Average & Bootstrapped",
#           fontsize=20, family='serif')
# plt.subplot(212)
# plt.plot(lls.ALL_PEAK_ARRAY, c.peaks, 'o', color='black')
# plt.errorbar(peak_wls_bs, c.peaks, xerr=c.wavelength_error(c.peaks),
#              fmt='.', color='green')
# plt.xlabel("Wavelength (A)", fontsize=16, family='serif')
# plt.ylabel("Pixel", fontsize=16, family='serif')
# plt.title("Calibrated Centroids versus Known Peaks at those Centroids",
#           fontsize=20, family='serif')
pltt.full_screen()

pltt.show()
