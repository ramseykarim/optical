import numpy as np
import matplotlib.pyplot as plt
import calibration as cal


class Sun:
    def __init__(self):
        self.calibrator = cal.Calibration()
        self.suns = self.calibrator.u.get_sun()
        self.center = 0
        self.curve = np.array([])
        self.light_curve()

    def light_curve(self):
        for frame in self.suns:
            self.curve = np.append(self.curve, np.sum(frame))
        self.center = np.argmax(self.curve)

    def calibrate_sun(self, sun_frame):
        wavelengths, spectrum = self.calibrator.calibrate(sun_frame)
        return wavelengths, spectrum

    def test_suns(self):
        self.generate_light_noise()
#        wavelengths, spectrum = self.calibrate_sun(self.suns[self.center])

#        plt.plot(wavelengths, spectrum, '.', color='orange')

    def generate_light_noise(self):
        background_indices = np.where(self.curve <= np.median(self.curve))
        specs = []
        wl = None
        for i in background_indices[0]:
            wl, spec = self.calibrator.calibrate(self.suns[i])
            specs.append(spec)
        plt.figure()
        for s in specs:
            plt.plot(wl, s, '.')
        specs = np.array(specs)
        spec = np.mean(specs, axis=0)
        plt.figure()
        plt.plot(wl, spec, '.', color='orange')
