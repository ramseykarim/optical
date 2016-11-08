import numpy as np
import matplotlib.pyplot as plt
import filefunctions as gsf
import star
import sys

HD189733_PATH = "/home/data/Planet_Transit/HD189733/"
HD189733_SCI_PATH = HD189733_PATH + "SI/"
HD189733_DRK_PATH = HD189733_PATH + "darks/dark_3s/"
HD189733_NAME = "HD189733 b"
HAT_PATH = "/home/data/Planet_Transit/HAT-P-56/"
HAT_SCI_PATH = HAT_PATH + "SI/"
HAT_DRK_PATH = HAT_PATH + "darks120s/"
HAT_NAME = "HAT-P-56 b"
RESPONSE_PATH = "ResponseMaps/"

STAR_PICKS = {}
FIRST_FRAME = np.array([])

JD = [2457683.83540622, 2457683.83694997, 2457683.84003919, 2457683.84159099,
      2457683.84313789, 2457683.84468276, 2457683.84777048, 2457683.84931572,
      2457683.85092093, 2457683.85401038, 2457683.85555526, 2457683.85709875,
      2457683.85864299, 2457683.8617346, 2457683.86328012, 2457683.86483095,
      2457683.86791958, 2457683.86946423, 2457683.87100829, 2457683.87255258,
      2457683.87564236, 2457683.87721155, 2457683.87875735, 2457683.88030666,
      2457683.88344576, 2457683.88499481, 2457683.88653833, 2457683.88963589,
      2457683.89118139, 2457683.89273533, 2457683.89583384, 2457683.89738136,
      2457683.89892472, 2457683.90201308, 2457683.90355898, 2457683.90510481,
      2457683.90665007, 2457683.90973955, 2457683.91128641, 2457683.91288379,
      2457683.91597179, 2457683.91751635, 2457683.91906319, 2457683.92061262,
      2457683.92370254, 2457683.92524755, 2457683.92679198, 2457683.92988234,
      2457683.93142968, 2457683.932974, 2457683.93451824, 2457683.93760868,
      2457683.93915835, 2457683.94070205, 2457683.94396299, 2457683.94551055,
      2457683.94705439, 2457683.9486005, 2457683.95169121, 2457683.95324015,
      2457683.9547852, 2457683.95787731, 2457683.95942187, 2457683.9609678,
      2457683.96251331, 2457683.96560509, 2457683.96715205, 2457683.96869805,
      2457683.97178872, 2457683.97333407, 2457683.97487791, 2457683.97642382,
      2457683.97796939, 2457683.97951367]

file_names = gsf.generate_names(HAT_SCI_PATH)
data = gsf.fits_open(file_names[0])


def jd_to_hours(jd_list):
    jd_array = np.array(jd_list)
    jd_array -= jd_array[0]
    jd_array *= 24.
    return jd_array


def median_subtract(science_frame):
    median = np.median(science_frame)
    return science_frame - median


def find_star(science_frame,
              (initial_x, initial_y),
              search_radius=20, coarse_radius=10, fine_radius=25):
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
    # return name_list


def filter_image(image):
    image[np.where((image < 1.))] = 1.
    return image


def filter_numbers(number):
    bad_list = [10, 46, 160]
    # if number in bad_list:
    #     return False
    # else:
    #     return True
    return True


# noinspection PyBroadException
class Tracking:
    def __init__(self, sci_path=HAT_SCI_PATH, dark_path=HAT_DRK_PATH,
                 band='v', name=HAT_NAME):
        self.planet_name = name
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
        return filter_image(median_subtract(self.flatten(self.dark_subtract(science_frame))))

    def find_initial(self, identifier, science_frame,
                     (initial_x, initial_y)):
        star_instance = star.Star(identifier)
        star_instance.add_frame(find_star(science_frame, (initial_x, initial_y)))
        self.stars[identifier] = star_instance

    def find_again(self, identifier, science_frame):
        star_instance = self.stars[identifier]
        star_instance.add_frame(find_star(science_frame, star_instance.last_seen))

    def find_again_hail_mary(self, identifier, science_frame):
        location = {'loc': (-99, -99), 'name': 0}

        def onclick(event):
            location['loc'] = (round(event.ydata), round(event.xdata))

        plt.ion()
        fig = plt.figure(9)
        ax = fig.add_subplot(111)
        ax.imshow(science_frame, cmap='Greys',
                  vmin=np.mean(FIRST_FRAME), vmax=2.5 * np.mean(FIRST_FRAME))
        fig.suptitle("Original Image")
        fig = plt.figure(10)
        ax = fig.add_subplot(111)
        ax.imshow(science_frame, cmap='Greys',
                  vmin=np.mean(science_frame), vmax=2.5 * np.mean(science_frame))
        fig.suptitle("Current Image")
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        print "User input requested for finding " + identifier
        print "Star was last seen"
        print "Original picks were"
        for s in STAR_PICKS:
            print s, STAR_PICKS[s]
        print "Please select and THEN name the same star. If the frame seems dead, write 'dead'."
        done = False
        a = "not set"
        while not done:
            a = raw_input("Name> ")
            print "You selected " + a + " to be located at " + str(location['loc']) + ". Is this ok? (yes/no)"
            done = True if raw_input("> ").lower() == 'yes' else False
        fig.canvas.mpl_disconnect(cid)
        plt.ioff()
        plt.close()
        plt.close()
        if a.lower() == 'dead':
            self.find_again_settle_for_less(identifier)
        else:
            star_instance = self.stars[identifier]
            star_instance.add_frame(find_star(science_frame, location['loc']))

    def find_again_settle_for_less(self, identifier):
        star_instance = self.stars[identifier]
        star_instance.duplicate_last()

    def star_search_loop(self, id_dictionary, aperture):
        for s in id_dictionary:
            self.find_initial(s, FIRST_FRAME, id_dictionary[s])
        aperture['apt'] = self.get_aperture(id_dictionary['science'])
        print "Aperture set to", aperture['apt']
        limit = len(self.file_names)
        for i in range(1, limit):
            # if not filter_numbers(i):
            #     continue
            sys.stdout.write("  " + str(int(i * 100. / (limit - 1))) + "%\r")
            sys.stdout.flush()
            science_frame = self.file_grab(i)
            science_frame = self.make_adjustments(science_frame)
            for s in id_dictionary:
                try:
                    self.find_again(s, science_frame)
                except:
                    # self.find_again_hail_mary(s, science_frame)
                    self.find_again_settle_for_less(s)
                    # plt.figure(5)
                    # plt.savefig("debug/save_"+'{num:02d}'.format(num=i) + ".png", bbox_inches='tight')
                    # gc.collect()
        print "\nDone!"

    def get_aperture(self, (initial_x, initial_y)):
        science_frame = self.file_grab(0)
        science_frame = self.make_adjustments(science_frame)
        self.find_initial('science', science_frame, (initial_x, initial_y))
        return self.stars['science'].get_aperture(0)

    def master_loop(self):
        identifier = {'id': -99}
        aperture = {'apt': -99}

        def onclick(event):
            STAR_PICKS[identifier['id']] = (round(event.ydata), round(event.xdata))

        global FIRST_FRAME
        FIRST_FRAME = self.file_grab(0)
        FIRST_FRAME = self.make_adjustments(FIRST_FRAME)
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(FIRST_FRAME, cmap='Greys',
                  vmin=np.mean(FIRST_FRAME), vmax=2.5 * np.mean(FIRST_FRAME))
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
        print "You selected " + str(len(STAR_PICKS)) + " targets:"
        for s in STAR_PICKS:
            print s, STAR_PICKS[s]
        if raw_input("Ready?").lower() == 'no':
            print "Aborting"
            sys.exit(0)
        print "Beginning reduction"
        # plt.ion()
        self.star_search_loop(STAR_PICKS, aperture)
        # plt.ioff()
        # plt.close()
        lc, lc_e = light_curve(self.stars, aperture['apt'])
        time = jd_to_hours(JD)
        plt.errorbar(time, lc, yerr=lc_e, fmt='.', color='k')
        plt.plot(time, np.ones(len(time)), '--', color='g')
        min_lc = np.where(lc == np.min(lc))[0]
        depth = np.median(lc[min_lc - 4:min_lc + 4])
        plt.plot(time, np.ones(len(time)) * depth, '--', color='g')
        plt.xlabel("Time (hours)")
        plt.ylabel("$\\langle R_{S} / R_{Ref} \\rangle$")
        plt.title("Normalized Light Curve of " + self.planet_name
                  + " || Depth: " + str(round((1 - depth) * 100, 2)) + "%")
        plt.savefig(self.planet_name + "_lc.pdf", bbox_inches='tight')
        plt.show()


def light_curve(stars, aperture):
    """
    Calculates averaged, normalized light curve WRT arbitrary number of reference stars
    :param aperture: integer aperture for integration
    :param stars: dictionary from strings to Star objects.
    Note that one key MUST be named 'science'!
    :return: Averaged light curve, Errors
    """
    num_refs = len(stars) - 1
    for s in stars:
        stars[s].calculate_power(aperture)
    science_p = stars['science'].power_array
    science_e = stars['science'].error_array
    ref_power = []
    ref_error = []
    for s in stars:
        if s == 'science':
            continue
        ref_power.append(stars[s].power_array)
        ref_error.append(stars[s].error_array)
    ratios = []
    ratio_errors = []
    snr_sci = (science_e / science_p) ** 2.
    for i in range(num_refs):
        ratios.append(science_p / ref_power[i])
        snr_ref = (ref_error[i] / ref_power[i]) ** 2.
        ratio_errors.append(np.sqrt(snr_sci + snr_ref) * ratios[i])
        ratios[i], ratio_errors[i] = norm_light_curve(ratios[i], ratio_errors[i])
    inverse_error_sum = sum([1 / (e ** 2.) for e in ratio_errors])
    lc = sum([r / (e ** 2.) for r, e in zip(ratios, ratio_errors)]) / inverse_error_sum
    lc_e = np.sqrt(sum([1 / e for e in ratio_errors]) / inverse_error_sum)
    # norm = np.mean(lc[len(lc) - 12:])
    # lc, lc_e = lc / norm, lc_e / norm
    return lc, lc_e


def norm_light_curve(lc, lc_e):
    norm = np.median(np.concatenate([lc[:10], lc[len(lc) - 10:]]))
    n_lc, n_lc_e = lc / norm, lc_e / norm
    return n_lc, n_lc_e


t = Tracking()
t.master_loop()
