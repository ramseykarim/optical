import numpy as np
import matplotlib.pyplot as plt
import unpacking as up


class Sun:
    def __init__(self):
        self.u = up.Unpack()

    def light_curve(self):
        curve = []
        suns = self.u.get_sun()
        for frame in suns:
            curve.append(np.sum(frame))
        plt.plot(curve, 'o')
        plt.xlabel("Frame number")
        plt.ylabel("Average Intensity (ADU)")
        plt.title("Solar \"light curve\"")
