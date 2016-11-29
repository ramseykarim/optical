import numpy as np
import matplotlib.pyplot as plt
import calibration as cal


class Sun:
    def __init__(self):
        self.calibrator = cal.Calibration()
        self.suns = self.calibrator.u.get_sun()
        self.center = 0

    def light_curve(self):
        curve = []
        for frame in self.suns:
            curve.append(np.sum(frame))
        self.center = np.argmax(np.array(curve))
        plt.plot(curve, 'o')
        plt.xlabel("Frame number")
        plt.ylabel("Average Intensity (ADU)")
        plt.title("Solar \"light curve\"")

    def calibrate_sun(self, sun_frame):
        wavelengths, spectrum = self.calibrator.calibrate(sun_frame)
        return wavelengths, spectrum

    def test_suns(self):
        wavelengths, spectrum = self.calibrate_sun(self.suns[self.center])
        plt.plot(wavelengths, spectrum, '.', color='orange')
