import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
from glob import glob
import os

fits_path = "/home/rkarim/flatfields/"
flat_path = "_band/*.fts"
dark_path = "_band_darks/*.fts"


def unpack(band_name):
    return_list = np.zeros([1336, 2004])
    counter = 0
    for f in glob(fits_path + band_name):
        print '.',
        hdu_list = pf.open(f)
        return_list += hdu_list[0].data
        hdu_list.close()
        counter += 1
    print '!'
    return return_list / counter


def produce_response(band):
    band_list = unpack(band + flat_path)
    band_dark_list = unpack(band + dark_path)
    band_field = band_list - band_dark_list
    band_hdu = pf.PrimaryHDU(band_field)
    band_hdu.writeto(band + "_band_response_map.fts")
    return band_field


def recover_response(band):
    hdu = pf.open(band + "_band_response_map.fts")[0]
    return hdu.data


r_band_field = recover_response('r')
v_band_field = recover_response('v')

color_map = "gist_earth"

plt.figure(1)
plt.imshow(r_band_field, cmap=color_map)
plt.title("R Band")
plt.colorbar()
plt.figure(2)
plt.imshow(v_band_field, cmap=color_map)
plt.title("V Band")
plt.colorbar()
plt.show()

