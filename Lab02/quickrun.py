import unpack as up
import calibration as cal
import numpy as np
import matplotlib.pyplot as plt

c = cal.Calibrate(up.Unpack())
all_wavelengths = np.array(cal.MERCURY_PEAKS + cal.NEON_PEAKS)
pixels = np.arange(2048)

deg1 = c.wavelength_fit(degree=1)
deg2 = c.wavelength_fit(degree=2)
deg3 = c.wavelength_fit(degree=3)

deg1comp = all_wavelengths - np.array([deg1[x] for x in c.peaks])
deg2comp = all_wavelengths - np.array([deg2[x] for x in c.peaks])
deg3comp = all_wavelengths - np.array([deg3[x] for x in c.peaks])

plt.figure(1)
cal.plot_peaks(c.neon, c.neon_centroids)
cal.plot_peaks(c.mercury, c.mercury_centroids)
plt.xlabel("Pixel Number")
plt.ylabel("Intensity")
plt.title("Measured Spectra for Ne and Hg with peaks")

plt.figure(2)

plt.subplot(221)
plt.plot(c.neon_centroids, cal.NEON_PEAKS, 'o', color='blue')
plt.plot(c.mercury_centroids, cal.MERCURY_PEAKS, 'o', color='blue')
plt.plot(pixels, deg1, '--')
plt.plot(pixels, deg2, '-')
plt.plot(pixels, deg3, '--')
plt.xlabel("Pixel")
plt.ylabel("Wavelength (A)")
plt.title("Wavelength Solution to Spectrometer")

plt.subplot(222)
plt.plot(c.peaks, deg1comp, 'o', color='green')
plt.xlabel("Pixel")
plt.ylabel("Difference (A)")
plt.title("1st Degree fit comparison")

plt.subplot(223)
plt.plot(c.peaks, deg2comp, 'o', color='green')
plt.xlabel("Pixel")
plt.ylabel("Difference (A)")
plt.title("2nd Degree fit comparison")
plt.tight_layout()

plt.subplot(224)
plt.plot(c.peaks, deg3comp, 'o', color='green')
plt.xlabel("Pixel")
plt.ylabel("Difference (A)")
plt.title("3rd Degree fit comparison")

plt.tight_layout()

plt.show()
