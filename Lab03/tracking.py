import numpy as np
import matplotlib.pyplot as plt
import filefunctions as gsf
import star
import gc
import sys


HAT_PATH = "/home/data/Planet_Transit/HAT-P-56/"
HAT_SCI_PATH = HAT_PATH + "SI/"
HAT_DRK_PATH = HAT_PATH + "darks120s/"
RESPONSE_PATH = "ResponseMaps/"

file_names = gsf.generate_names(HAT_SCI_PATH)
data = gsf.fits_open(file_names[0])


def median_subtract(science_frame):
    median = np.median(science_frame)
    return science_frame - median


def find_star(science_frame,
              (initial_x, initial_y),
              search_radius=50, coarse_radius=20, fine_radius=50):
    x_t, y_t, x, y, star_box = star.find_centroid_in_range(science_frame,
                                                           (initial_x, initial_y),
                                                           search_radius, coarse_radius,
                                                           fine_radius)
    return x_t, y_t, x, y, star_box


def filter_names(name_list):
    new_list = []
    bad_list = ['0002', '0007', '0011', '0016',
                '0020', '0025', '0030', '0034',
                '0038', '0042', '0047', '0051',
                '0056', '0060', '0065', '0069',
                '0074', '0078', '0083', '0087']
    for n in name_list:
        good = True
        for m in bad_list:
            if m in n:
                good = False
        if good:
            new_list.append(n)
    return new_list


class Tracking:
    def __init__(self, sci_path=HAT_SCI_PATH, dark_path=HAT_DRK_PATH, band='v'):
        self.sci_path = sci_path
        self.dark_frame = gsf.process_dark(dark_path)
        self.response_map = gsf.fits_open(RESPONSE_PATH + band.lower() + "_band_response_map.fts")
        self.stars = {}
        self.file_names = filter_names(gsf.generate_names(HAT_SCI_PATH))
        self.aperture = -1

    def file_grab(self, index):
        return gsf.fits_open(self.file_names[index])

    def dark_subtract(self, science_frame):
        return science_frame - self.dark_frame

    def flatten(self, science_frame):
        return science_frame / self.response_map

    def make_adjustments(self, science_frame):
        return median_subtract(self.flatten(self.dark_subtract(science_frame)))

    def find_initial(self, identifier, science_frame,
                     (initial_x, initial_y)):
        star_instance = star.Star(identifier)
        star_instance.add_frame(find_star(science_frame, (initial_x, initial_y)))
        self.stars[identifier] = star_instance

    def find_again(self, identifier, science_frame):
        star_instance = self.stars[identifier]
        star_instance.add_frame(find_star(science_frame, star_instance.last_seen))

    def star_search_loop(self, id_dictionary, aperture):
        science_frame = self.file_grab(0)
        science_frame = self.make_adjustments(science_frame)
        for s in id_dictionary:
            self.find_initial(s, science_frame, id_dictionary[s])
        aperture['apt'] = self.get_aperture(id_dictionary['science'])
        print "Aperture set to", aperture['apt']
        limit = len(self.file_names)
        for i in range(1, limit):
            sys.stdout.write(str(int(i*100./(limit - 1))) + " %\r")
            sys.stdout.flush()
            science_frame = self.file_grab(i)
            science_frame = self.make_adjustments(science_frame)
            for s in id_dictionary:
                try:
                    self.find_again(s, science_frame)
                except IndexError:
                    print self.file_names[i]
            gc.collect()
        print "Done!"

    def get_aperture(self, (initial_x, initial_y)):
        science_frame = self.file_grab(0)
        science_frame = self.make_adjustments(science_frame)
        self.find_initial('science', science_frame, (initial_x, initial_y))
        return self.stars['science'].get_aperture(0)

    def master_loop(self):
        star_picks = {}
        identifier = {'id': -99}
        aperture = {'apt': -99}

        def onclick(event):
            star_picks[identifier['id']] = (round(event.ydata), round(event.xdata))

        first_frame = self.file_grab(0)
        first_frame = self.make_adjustments(first_frame)
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(first_frame, cmap='Greys')
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        print "Name a star and then click it. Enter 'd' when done."
        print "PLEASE name at least one star 'science'!"
        done = False
        while not done:
            a = raw_input("Name> ")
            if a == 'd':
                done = True
                continue
            identifier['id'] = a
            print "Now click the star."
            print "Name and select further stars."
        fig.canvas.mpl_disconnect(cid)
        plt.ioff()
        plt.close()
        print "You selected " + str(len(star_picks)) + " targets:"
        for s in star_picks:
            print s, star_picks[s]
        if raw_input("Ready?").lower() == 'no':
            print "Aborting"
            sys.exit(0)
        print "Beginning reduction"
        self.star_search_loop(star_picks, aperture)
        for s in self.stars:
            self.stars[s].calculate_power(aperture['apt'])
        first = np.array(self.stars['science'].power_array)
        second = np.array(self.stars['ref1'].power_array)
        third = np.array(self.stars['ref2'].power_array)
        plt.figure()
        plt.subplot(131)
        plt.title("F/S")
        plt.plot(first/second)

        plt.subplot(132)
        plt.title("S/T")
        plt.plot(second/third)

        plt.subplot(133)
        plt.title("F/T")
        plt.plot(first/third)

        plt.figure()
        plt.subplot(131)
        plt.title("first")
        plt.plot(first)

        plt.subplot(132)
        plt.title("second")
        plt.plot(second)

        plt.subplot(133)
        plt.title("third")
        plt.plot(third)

        plt.show()


t = Tracking()
t.master_loop()
