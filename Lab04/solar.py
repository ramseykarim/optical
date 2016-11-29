import numpy as np
import matplotlib.pyplot as plt
import calibration as cal


class Sun:
    def __init__(self):
        self.c = cal.Calibration()
        self.u = self.c.u

    def light_curve(self):
        curve = []
        suns = self.u.get_sun()
        for frame in suns:
            curve.append(np.sum(frame))
        plt.plot(curve, 'o')
        plt.xlabel("Frame number")
        plt.ylabel("Average Intensity (ADU)")
        plt.title("Solar \"light curve\"")

    def calibrate_sun(self, sun_1d_orders):
        """
        Expects output of integrate_sun
        :param sun_1d_orders: This should be the list of integrals for each order,
        as is output by Sun.integrate_sun(sun_frame)
        :return: Single 1D array of wavelengths, spectrum in order of increasing wavelength.
        Has already been interpolated.
        """

    def integrate_sun(self, sun_frame):
        """

        :param sun_frame:
        :return:
        """
        integrals = [cal.integrate(sun_frame, order) for order in self.flat_map]
        return integrals
