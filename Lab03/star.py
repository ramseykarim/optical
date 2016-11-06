import numpy as np
import matplotlib.pyplot as plt


class Star:
    def __init__(self, identifier):
        self.identifier = identifier
        self.frames = []
        self.last_seen = (-1, -1)

    def add_frame(self, x_y_img_tuple):
        assert isinstance(x_y_img_tuple, tuple)
        frame = Frame(x_y_img_tuple)
        self.frames.append(frame)
        self.last_seen = (frame.x_entire, frame.y_entire)

    def get_aperture(self, index):
        find_aperture(self.frames[index])


def find_aperture(frame):
    assert isinstance(frame, Frame)
    find_background_noise(frame.image)


def find_background_noise(image):
    assert isinstance(image, np.ndarray)
    dim_1 = image.shape[0]
    dim_2 = image.shape[1]
    print dim_1, dim_2
    side_1 = image[:3, :].flatten()
    side_2 = image[dim_1 - 4:, :].flatten()
    side_3 = image[3:dim_1 - 4, :3].flatten()
    side_4 = image[3:dim_1 - 4, dim_2 - 4:].flatten()
    noise_array = np.concatenate([side_1, side_2, side_3, side_4])


def find_centroid_in_range(science_frame, (initial_x, initial_y),
                           search_radius, coarse_radius, fine_radius):
    x_c, y_c, x_c_1, y_c_1 = find_centroid_helper(science_frame, (initial_x, initial_y), search_radius)
    x_c_f, y_c_f, x_c_1, y_c_1 = find_centroid_helper(science_frame, (x_c, y_c), coarse_radius)
    star_box, lo_x, lo_y = image_partition(science_frame, (x_c_f, y_c_f), fine_radius)
    return x_c_f, y_c_f, x_c_1, y_c_1, star_box


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
    return np.sum(x * y) / np.sum(y)


def centroid_2d(box):
    x_centroid = centroid(np.sum(box, axis=1))
    y_centroid = centroid(np.sum(box, axis=0))
    return x_centroid, y_centroid


class Frame:
    def __init__(self, (x_entire, y_entire, x, y, image)):
        print "MEDIAN of IMG", np.median(image)

        self.x_entire = x_entire
        self.y_entire = y_entire
        self.x = x
        self.y = y
        self.image = image
