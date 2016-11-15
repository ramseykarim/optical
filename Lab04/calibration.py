import numpy as np
import unpacking as up
import matplotlib.pyplot as plt


def boolean_array(array, std_multiplier):
    thresh_hold = np.mean(array) + np.std(array) * std_multiplier
    print thresh_hold
    ones = np.where(array > thresh_hold)
    new_array = np.zeros(array.shape)
    new_array[ones] += 1
    return new_array


def integrate(array, where_list):
    integral = np.array([])
    for where in where_list:
        integral = np.append(integral, np.sum(array[where]))
    return integral


class Calibration:
    def __init__(self):
        self.u = up.Unpack()

    def examine_flat(self):
        flat = np.zeros([1024, 1048])
        flat_map = self.spec_map()
        for where in flat_map:
            flat[where] += 1
        plt.imshow(flat)
        plt.colorbar()
        plt.show()

    def integrate_neon(self):
        neon = self.u.get_neon()
        integral = integrate(neon, self.spec_map())
        plt.plot(integral)
        plt.show()

    def spec_map(self):
        flat = boolean_array(self.u.get_halogen(), 1)
        y, x = flat.shape
        print y
        print x
        step_size = 30
        padding = 20
        i = -1 * step_size
        integration_map = []
        count = 0
        doit = False
        while i < y - step_size:
            ul, ur = flat[i, padding], flat[i, x - padding]
            ll, lr = flat[i + step_size, padding], flat[i + step_size, x - padding]
            if ul + ur + ll + lr == 0:
                l, r = np.sum(flat[i:i+step_size, padding]), np.sum(flat[i:i+step_size, x - padding])
                if l > 0 and r > 0:
                    count += 1
                    for j in range(padding, x - padding, 1):
                        int_y, int_x = np.where(flat[i:i+step_size, j:j+1] == 1)
                        integrand = (int_y + i, int_x + j)
                        integration_map.append(integrand)
                    i += step_size
            i += step_size / 2
        print "---"
        print count
        print "---"
        return integration_map
