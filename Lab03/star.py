import numpy as np
import matplotlib.pyplot as plt
import sys

subplot_mgmt = {'science': 221, 'ref1': 222,
                'ref2': 223, 'ref3': 224}


class Star:
    def __init__(self, identifier):
        self.identifier = identifier
        self.frames = []
        self.last_seen = (-1, -1)
        self.power_array = -99
        self.error_array = -99

    def add_frame(self, x_y_img_tuple):
        assert isinstance(x_y_img_tuple, tuple)
        frame = Frame(x_y_img_tuple)
        self.frames.append(frame)
        self.last_seen = (frame.x_entire, frame.y_entire)
        # plt.figure(5)
        # plt.subplot(subplot_mgmt[self.identifier])
        # plt.imshow(frame.image)
        # plt.title(self.identifier)
        # plt.draw()

    def duplicate_last(self):
        self.frames.append(self.frames[len(self.frames) - 1].copy())

    def get_aperture(self, index):
        apt = find_aperture(self.frames[index])
        return apt if apt < 15 else 15
        # return 7

    def calculate_power(self, aperture):
        print "Integrating star " + self.identifier.upper() + "...",
        for f in self.frames:
            f.get_total_power(aperture)
        print "done"
        self.power_array = np.array([f.total_power for f in self.frames])
        self.error_array = np.array([f.power_error for f in self.frames])


class Frame:
    def __init__(self, (x_entire, y_entire, x, y, image)):
        self.x_entire = x_entire
        self.y_entire = y_entire
        self.x = x
        self.y = y
        self.image = image
        self.total_power = -99
        self.power_error = -99

    def get_total_power(self, radius=9):
        self.total_power, self.power_error = integrate_aperture(self, radius)

    def copy(self):
        return Frame((self.x_entire, self.y_entire,
                      self.x, self.y, self.image))

"""
APERTURE
"""


def integrate_aperture(frame, radius):
    x, y = frame.x, frame.y
    adu = 0
    pixels = 0
    for i in range(int(x) - radius + 1, int(x) + radius - 1):
        for j in range(int(y) - radius + 1, int(y) + radius - 1):
            distance = np.round(np.sqrt((i - x) ** 2. + (j - y) ** 2.))
            if distance < radius:
                adu += frame.image[i, j]
                pixels += 1
    error = noise(frame.image, adu, pixels)
    return adu, error


def find_aperture(frame):
    assert isinstance(frame, Frame)
    x, y = frame.x, frame.y
    radius = frame.image.shape[0] / 2
    adu_by_radius = np.zeros(radius)
    pixels_by_radius = np.zeros(radius)
    for i in range(int(x) - radius + 1, int(x) + radius - 1):
        for j in range(int(y) - radius + 1, int(y) + radius - 1):
            distance = np.round(np.sqrt((i - x)**2. + (j - y)**2.))
            if distance < radius:
                pixel_array = np.ones(radius - distance)
                adu_array = pixel_array * frame.image[i, j]
                zero_array = np.zeros(distance)
                adu_by_radius += np.concatenate([zero_array, adu_array])
                pixels_by_radius += np.concatenate([zero_array, pixel_array])
    sigma_f = noise(frame.image, adu_by_radius, pixels_by_radius)
    signal_noise = adu_by_radius / sigma_f
    aperture = np.where(signal_noise == np.max(signal_noise))[0][0]
    # print "DESIRED APERTURE:", aperture
    # plt.figure()
    # plt.plot(signal_noise)
    # plt.plot([aperture, aperture], [np.min(signal_noise), np.max(signal_noise)])
    # plt.show()
    return aperture


def noise(image, adu, pixels):
    sigma_b = find_background_noise(image)
    n_sigma_b_sq = background_noise_sq(pixels, sigma_b)
    sigma_p_sq = pixel_noise_sq(adu)
    sigma_f = np.sqrt(sigma_p_sq + n_sigma_b_sq)
    return sigma_f


def pixel_noise_sq(adu):
    n_photon = adu * 1.7 / 0.37
    sigma_p = np.sqrt(n_photon) * 0.37 / 1.7
    return sigma_p ** 2.


def background_noise_sq(pixels, background_noise):
    return pixels * (background_noise ** 2.)


def find_background_noise(image):
    assert isinstance(image, np.ndarray)
    dim_1 = image.shape[0]
    dim_2 = image.shape[1]
    side_1 = image[:5, :].flatten()
    side_2 = image[dim_1 - 6:, :].flatten()
    side_3 = image[5:dim_1 - 6, :5].flatten()
    side_4 = image[5:dim_1 - 6, dim_2 - 6:].flatten()
    noise_array = np.concatenate([side_1, side_2, side_3, side_4])
    total_noise = np.std(noise_array)
    return total_noise


def find_centroid_in_range(science_frame, (initial_x, initial_y),
                           search_radius, coarse_radius, fine_radius):
    x_guess, y_guess = max_2d(science_frame, (initial_x, initial_y), search_radius)
    x_c, y_c, x_c_1, y_c_1 = find_centroid_helper(science_frame, (x_guess, y_guess), search_radius)
    x_c_f, y_c_f, x_c_1, y_c_1 = find_centroid_helper(science_frame, (x_c, y_c), coarse_radius)
    star_box, lo_x, lo_y = image_partition(science_frame, (x_c_f, y_c_f), fine_radius)
    return x_c_f, y_c_f, x_c_f - lo_x, y_c_f - lo_y, star_box


def find_centroid_helper(frame, (initial_x, initial_y), radius):
    search_box, lo_x, lo_y = image_partition(frame, (initial_x, initial_y), radius)
    x_c, y_c = centroid_2d(search_box)
    x_c_entire, y_c_entire = x_c + lo_x, y_c + lo_y
    return x_c_entire, y_c_entire, x_c, y_c


def image_partition(frame, (initial_x, initial_y), radius):
    lo_x, hi_x = initial_x - radius, initial_x + radius
    lo_y, hi_y = initial_y - radius, initial_y + radius
    search_box = frame[lo_x:hi_x, lo_y:hi_y]
    return search_box, lo_x, lo_y


def centroid(y):
    x = np.arange(len(y))
    cent = np.nansum(x * y) / np.nansum(y)
    return cent


def centroid_2d(box):
    x_centroid = centroid(np.nansum(box, axis=1))
    y_centroid = centroid(np.nansum(box, axis=0))
    return x_centroid, y_centroid


def max_2d(image, (initial_x, initial_y), radius):
    box, lo_x, lo_y = image_partition(image, (initial_x, initial_y), radius)
    x = np.nansum(box, axis=1)
    y = np.nansum(box, axis=0)
    x = np.where(x == np.max(x))[0]
    y = np.where(y == np.max(y))[0]
    return x + lo_x, y + lo_y

